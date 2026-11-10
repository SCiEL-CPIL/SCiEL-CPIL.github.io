import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 55

    # Precompute all 3-APs: list of (a, a+d, a+2d)
    three_aps = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    # Precompute all 4-APs: list of (a, a+d, a+2d, a+3d)
    four_aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each element x, precompute:
    # - three_ap_partners[x]: list of (y, z) such that {x,y,z} forms a 3-AP
    # - four_ap_conflicts[x]: list of 4-AP indices that contain x
    # - four_ap_members[ap_idx]: the 4 elements
    
    # Build mapping: element -> list of 3-AP indices it participates in
    elem_to_3aps = [[] for _ in range(n + 1)]
    for i, (a, b, c) in enumerate(three_aps):
        elem_to_3aps[a].append(i)
        elem_to_3aps[b].append(i)
        elem_to_3aps[c].append(i)
    
    # Build mapping: element -> list of 4-AP indices it participates in
    elem_to_4aps = [[] for _ in range(n + 1)]
    for i, (a, b, c, d) in enumerate(four_aps):
        elem_to_4aps[a].append(i)
        elem_to_4aps[b].append(i)
        elem_to_4aps[c].append(i)
        elem_to_4aps[d].append(i)
    
    # For fast 4-AP conflict checking: given a set S (as bitmask), 
    # check if adding y creates a 4-AP.
    # Precompute for each y: the set of 4-APs that y is part of.
    # A 4-AP is "active" in S if all 4 elements are in S.
    # Adding y creates a 4-AP iff there's a 4-AP containing y where the other 3 are in S.
    
    # Use numpy bitmasks for speed
    # Represent set S as a numpy array of bools or as an integer bitmask
    # For n=256, we can use a Python int as bitmask (256 bits)
    
    def set_to_mask(S):
        m = 0
        for x in S:
            m |= (1 << (x - 1))
        return m
    
    def mask_to_set(m):
        S = set()
        for x in range(1, n + 1):
            if m & (1 << (x - 1)):
                S.add(x)
        return S
    
    # Precompute for each element y and each 4-AP containing y, 
    # the bitmask of the other 3 elements
    four_ap_other_mask = {}  # (y, ap_idx) -> bitmask of other 3 elements
    for i, (a, b, c, d) in enumerate(four_aps):
        for y in (a, b, c, d):
            others = [z for z in (a, b, c, d) if z != y]
            m = 0
            for z in others:
                m |= (1 << (z - 1))
            four_ap_other_mask[(y, i)] = m
    
    # Count 3-APs in a set S
    def count_3aps_in_set(S):
        S_set = set(S)
        count = 0
        for a, b, c in three_aps:
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count
    
    # Check if S has any 4-AP
    def has_4ap(S):
        S_set = set(S)
        for a, b, c, d in four_aps:
            if a in S_set and b in S_set and c in S_set and d in S_set:
                return True
        return False
    
    # Check if adding y to S creates a 4-AP (using precomputed masks)
    def adding_y_creates_4ap(y, S_mask):
        for ap_idx in elem_to_4aps[y]:
            other_mask = four_ap_other_mask[(y, ap_idx)]
            if (S_mask & other_mask) == other_mask:
                return True
        return False
    
    # Count 3-APs that y would form with S
    def gain_from_adding_y(y, S):
        S_set = set(S)
        count = 0
        for ap_idx in elem_to_3aps[y]:
            a, b, c = three_aps[ap_idx]
            others = [z for z in (a, b, c) if z != y]
            if all(z in S_set for z in others):
                count += 1
        return count
    
    # Lost 3-APs if removing y from S
    def loss_from_removing_y(y, S):
        S_set = set(S)
        count = 0
        for ap_idx in elem_to_3aps[y]:
            a, b, c = three_aps[ap_idx]
            others = [z for z in (a, b, c) if z != y]
            if all(z in S_set for z in others):
                count += 1
        return count
    
    # GRASP + Tabu search
    best_score = -1
    best_set = set()
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # GRASP: greedy construction with randomized ordering
        if restart == 0:
            # First restart: pure greedy (no randomness)
            order = list(range(1, n + 1))
        else:
            # Randomize with alpha parameter
            order = list(range(1, n + 1))
            random.shuffle(order)
        
        S = set()
        S_mask = 0
        
        # Greedy: add elements in order if they don't create 4-AP
        for x in order:
            if not adding_y_creates_4ap(x, S_mask):
                S.add(x)
                S_mask |= (1 << (x - 1))
        
        current_score = count_3aps_in_set(S)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(S)
        
        # Tabu search phase
        tabu_list = {}  # element -> iteration when it becomes non-tabu
        tabu_tenure = 15
        max_tabu_iters = 300
        no_improve_count = 0
        max_no_improve = 30
        
        for tabu_iter in range(max_tabu_iters):
            if time.time() > deadline - 3:
                break
            
            best_move_gain = -1
            best_move_type = None  # 'add', 'remove', 'swap'
            best_move_x = None  # for swap: element to remove
            best_move_y = None  # for add/swap: element to add
            
            # Try adds
            for y in range(1, n + 1):
                if y in S:
                    continue
                if tabu_list.get(y, 0) > tabu_iter:
                    continue
                if not adding_y_creates_4ap(y, S_mask):
                    gain = gain_from_adding_y(y, S)
                    if gain > best_move_gain:
                        best_move_gain = gain
                        best_move_type = 'add'
                        best_move_x = None
                        best_move_y = y
            
            # Try removes
            for x in S:
                if tabu_list.get(x, 0) > tabu_iter:
                    continue
                gain = -loss_from_removing_y(x, S)
                if gain > best_move_gain:
                    best_move_gain = gain
                    best_move_type = 'remove'
                    best_move_x = x
                    best_move_y = None
            
            # Try swaps (sampled for speed)
            if time.time() < deadline - 5:
                num_swap_samples = 50
                candidates_y = [y for y in range(1, n + 1) if y not in S]
                for _ in range(num_swap_samples):
                    x = random.choice(list(S))
                    if not candidates_y:
                        break
                    y = random.choice(candidates_y)
                    if tabu_list.get(x, 0) > tabu_iter or tabu_list.get(y, 0) > tabu_iter:
                        continue
                    
                    # Check if y can be added after removing x
                    temp_mask = S_mask & ~(1 << (x - 1))
                    if adding_y_creates_4ap(y, temp_mask):
                        continue
                    
                    # Calculate gain: lose 3-APs with x, gain 3-APs with y (minus shared)
                    loss_x = loss_from_removing_y(x, S)
                    gain_y_raw = gain_from_adding_y(y, S)
                    
                    # Subtract shared 3-APs (those containing both x and y)
                    shared = 0
                    S_set = set(S)
                    for ap_idx in elem_to_3aps[x]:
                        a, b, c = three_aps[ap_idx]
                        if y in (a, b, c):
                            # This 3-AP contains both x and y
                            others = [z for z in (a, b, c) if z not in (x, y)]
                            if all(z in S_set for z in others):
                                shared += 1
                    
                    gain = -loss_x + (gain_y_raw - shared)
                    
                    if gain > best_move_gain:
                        best_move_gain = gain
                        best_move_type = 'swap'
                        best_move_x = x
                        best_move_y = y
            
            if best_move_type is None or best_move_gain < 0:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
                continue
            
            # Execute best move
            if best_move_type == 'add':
                S.add(best_move_y)
                S_mask |= (1 << (best_move_y - 1))
                tabu_list[best_move_y] = tabu_iter + tabu_tenure
            elif best_move_type == 'remove':
                S.remove(best_move_x)
                S_mask &= ~(1 << (best_move_x - 1))
                tabu_list[best_move_x] = tabu_iter + tabu_tenure
            elif best_move_type == 'swap':
                S.remove(best_move_x)
                S.add(best_move_y)
                S_mask &= ~(1 << (best_move_x - 1))
                S_mask |= (1 << (best_move_y - 1))
                tabu_list[best_move_x] = tabu_iter + tabu_tenure
                tabu_list[best_move_y] = tabu_iter + tabu_tenure
            
            current_score += best_move_gain
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(S)
        
        # Verify best_set is valid
        if best_set and has_4ap(best_set):
            # Should not happen, but verify
            pass
    
    # Final verification
    if best_set and has_4ap(best_set):
        # Fallback to simple greedy
        S = set()
        for x in range(1, n + 1):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if all((x - j * d) in S for j in (1, 2, 3)):
                    valid = False
                    break
            if valid:
                S.add(x)
        best_set = S
        best_score = count_3aps_in_set(S)
    
    return sorted(best_set)
