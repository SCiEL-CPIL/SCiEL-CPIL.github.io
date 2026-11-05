import random
import time
import itertools
import math
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

def _is_addable(p, S_set):
    """Check if p can be added to S_set without creating an isosceles triangle."""
    if p in S_set:
        return False
    
    # Check 1: Distances from p to all points in S_set must be distinct.
    dists_from_p = set()
    for q in S_set:
        d = dist2[p, q]
        if d in dists_from_p:
            return False
        dists_from_p.add(d)
        
    # Check 2: For each q in S_set, d(p, q) must not equal d(q, r) for any r in S_set, r != p.
    for q in S_set:
        d_pq = dist2[p, q]
        for r in S_set:
            if r == p:
                continue
            if dist2[q, r] == d_pq:
                return False
                
    return True

def _get_addable(S_set):
    """Get set of points that can be added to S_set."""
    addable = set()
    for p in range(N_PTS):
        if p not in S_set:
            if _is_addable(p, S_set):
                addable.add(p)
    return addable

def _greedy_initialization(deadline):
    """Greedy construction of a maximal no-isosceles set using blocking power heuristic."""
    S = set()
    
    while time.time() < deadline:
        addable = _get_addable(S)
        if not addable:
            break
            
        # Choose the point that blocks the most other addable points
        best_p = -1
        best_block_count = -1
        
        for p in addable:
            new_S = S | {p}
            new_addable = _get_addable(new_S)
            blocked = len(addable) - len(new_addable)
            if blocked > best_block_count:
                best_block_count = blocked
                best_p = p
                
        if best_p == -1:
            break
            
        S.add(best_p)
        
    return S

def _local_search(S_initial, deadline):
    """
    Local search to minimize |S| while maintaining maximality and validity.
    Uses simulated annealing with removal-based moves.
    """
    S_set = set(S_initial)
    current_size = len(S_set)
    best_set = set(S_set)
    best_size = current_size
    
    temp = 15.0
    cooling = 0.995
    min_temp = 0.5
    no_improve_steps = 0
    max_no_improve = 300
    
    while time.time() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        if not S_set:
            break
            
        # Try to remove a random point
        p_out = random.choice(list(S_set))
        new_S_set = S_set - {p_out}
        
        # Find addable points for new_S_set
        addable = _get_addable(new_S_set)
        
        # If no addable points, we have a smaller maximal set
        if not addable:
            S_set = new_S_set
            current_size = len(S_set)
            if current_size < best_size:
                best_size = current_size
                best_set = set(S_set)
                no_improve_steps = 0
            continue
            
        # If there are addable points, we must add some back to maintain maximality
        # Choose the point that blocks the most other addable points
        best_p_in = -1
        best_block_count = -1
        
        # Limit the search to a few candidates to save time
        candidates = list(addable)
        random.shuffle(candidates)
        for p_in in candidates[:40]:
            new_S_set2 = new_S_set | {p_in}
            new_addable = _get_addable(new_S_set2)
            blocked = len(addable) - len(new_addable)
            if blocked > best_block_count:
                best_block_count = blocked
                best_p_in = p_in
                
        if best_p_in == -1:
            continue
            
        new_S_set = new_S_set | {best_p_in}
        new_size = len(new_S_set)
        
        delta_e = new_size - current_size
        if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
            S_set = new_S_set
            current_size = new_size
            if current_size < best_size:
                best_size = current_size
                best_set = set(S_set)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        # Kick if stuck
        if no_improve_steps > max_no_improve:
            # Randomly swap two points
            if len(S_set) > 1:
                p1, p2 = random.sample(list(S_set), 2)
                S_set.remove(p1)
                S_set.remove(p2)
                # Add two random addable points
                addable = _get_addable(S_set)
                if len(addable) >= 2:
                    new_p1, new_p2 = random.sample(list(addable), 2)
                    S_set.add(new_p1)
                    S_set.add(new_p2)
                    current_size = len(S_set)
            no_improve_steps = 0
            temp = max(temp, 10.0)
            
        temp *= cooling
        
    return list(best_set)

def _pair_swap_local_search(S_initial, deadline):
    """
    Alternative local search: try swapping pairs of points to reduce size or improve structure.
    """
    S_set = set(S_initial)
    current_size = len(S_set)
    best_set = set(S_set)
    best_size = current_size
    
    temp = 10.0
    cooling = 0.995
    no_improve_steps = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        if temp < 0.5:
            temp = 0.5
            
        if len(S_set) < 2:
            break
            
        # Try a random pair swap: remove p1, add p2
        p1, p2 = random.sample(list(S_set), 2)
        new_S_set = S_set - {p1}
        
        # Check if p2 can be added to new_S_set
        if _is_addable(p2, new_S_set):
            new_S_set.add(p2)
            new_size = len(new_S_set)
            
            # Check if the new set is still maximal (should be if we just swapped)
            # Actually, we need to verify maximality
            addable = _get_addable(new_S_set)
            if not addable:
                # Maximal and valid
                delta_e = new_size - current_size
                if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
                    S_set = new_S_set
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best_set = set(S_set)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
            else:
                # Not maximal, need to add points back
                # For now, skip this move
                no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        if no_improve_steps > max_no_improve:
            # Kick: remove a random point and re-greedy
            if S_set:
                p_out = random.choice(list(S_set))
                S_set.remove(p_out)
                # Re-greedy to fill up
                while time.time() < deadline:
                    addable = _get_addable(S_set)
                    if not addable:
                        break
                    # Pick the most blocking point
                    best_p = -1
                    best_block = -1
                    for p in addable:
                        new_S = S_set | {p}
                        new_addable = _get_addable(new_S)
                        blocked = len(addable) - len(new_addable)
                        if blocked > best_block:
                            best_block = blocked
                            best_p = p
                    if best_p == -1:
                        break
                    S_set.add(best_p)
                current_size = len(S_set)
            no_improve_steps = 0
            temp = max(temp, 5.0)
            
        temp *= cooling
        
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts with alternating local search strategies
    restart_count = 0
    while time.time() < deadline - 15.0:
        S_init = _greedy_initialization(deadline - 10.0)
        
        # Alternate between different local search strategies
        if restart_count % 3 == 0:
            S_opt = _local_search(S_init, deadline - 5.0)
        elif restart_count % 3 == 1:
            S_opt = _pair_swap_local_search(S_init, deadline - 5.0)
        else:
            # Combine: do local search then pair swap
            S_mid = _local_search(S_init, (start_time + deadline) // 2)
            S_opt = _pair_swap_local_search(S_mid, deadline - 5.0)
        
        if len(S_opt) < best_size:
            best_size = len(S_opt)
            best_set = S_opt
            
        restart_count += 1
        
    if best_set is None:
        best_set = _greedy_initialization(deadline)
        
    return [ALL_POINTS[i] for i in best_set]
