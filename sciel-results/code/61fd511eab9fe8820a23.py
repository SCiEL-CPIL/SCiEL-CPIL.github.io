import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) for d >= 1, a >= 1, a+3d <= N
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # Map each number to the indices of APs it belongs to
    ap_indices_by_num = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)
    
    # Precompute for each AP, the set of other elements (excluding each position)
    # For faster can_add_element checks
    
    def is_valid_set(S):
        """Check if set S contains any 4-AP"""
        S_set = set(S) if not isinstance(S, set) else S
        for ap in all_aps:
            if all(x in S_set for x in ap):
                return False
        return True
    
    def can_add_element(x, S):
        """Check if x can be added to set S without creating a 4-AP"""
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            count_in_set = 0
            for elem in ap:
                if elem == x:
                    continue
                if elem in S:
                    count_in_set += 1
            if count_in_set == 3:
                return False
        return True
    
    def greedy_with_order(order, S_init):
        """Greedy add elements in the given order to S_init"""
        S = set(S_init)
        for x in order:
            if x not in S and can_add_element(x, S):
                S.add(x)
        return S
    
    # Base-4 {0,1} construction (Salem-Spencer style)
    # Numbers whose base-4 representation contains only 0s and 1s
    # This set is 3-AP-free, hence also 4-AP-free
    base4_set = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_set.add(i)
    
    if not is_valid_set(base4_set):
        base4_set = set()
    
    # Initial best from various heuristics
    candidates_for_best = []
    
    # 1. Base-4 set
    candidates_for_best.append(base4_set)
    
    # 2. Greedy from low to high
    S_greedy_low = greedy_with_order(range(1, N + 1), set())
    candidates_for_best.append(S_greedy_low)
    
    # 3. Greedy from high to low
    S_greedy_high = greedy_with_order(range(N, 0, -1), set())
    candidates_for_best.append(S_greedy_high)
    
    # 4. Greedy starting from base-4 set, try to add more
    S_greedy_from_base4 = greedy_with_order(range(1, N + 1), set(base4_set))
    candidates_for_best.append(S_greedy_from_base4)
    
    # 5. Randomized greedy with different seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_rand = greedy_with_order(order, set())
        candidates_for_best.append(S_rand)
    
    best_set = max(candidates_for_best, key=len)
    best_size = len(best_set)
    
    # Local search: try to improve by removing one element and adding others
    for restart in range(5000):
        if time.time() > deadline - 5:
            break
            
        current_set = set(best_set)
        elements = list(current_set)
        random.shuffle(elements)
        
        for i in range(len(elements)):
            if time.time() > deadline - 5:
                break
            
            remove_elem = elements[i]
            current_set.remove(remove_elem)
            
            # Try to add candidates
            candidates = [j for j in range(1, N + 1) if j not in current_set]
            random.shuffle(candidates)
            
            added = []
            for c in candidates:
                if can_add_element(c, current_set):
                    current_set.add(c)
                    added.append(c)
            
            if len(added) > 1:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set
                break
            else:
                current_set.add(remove_elem)
                for a in added:
                    if a in current_set:
                        current_set.remove(a)
    
    # Second phase: more aggressive search with different strategies
    for restart in range(3000):
        if time.time() > deadline - 2:
            break
        
        # Start from best_set with random perturbations
        current_set = set(best_set)
        elements = list(current_set)
        random.shuffle(elements)
        
        # Remove 1-3 random elements
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            if elements:
                rem = elements.pop()
                current_set.remove(rem)
        
        # Try to add back and new elements
        candidates = [j for j in range(1, N + 1) if j not in current_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if can_add_element(c, current_set):
                current_set.add(c)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Third phase: try to find larger sets using simulated annealing style
    current_best = set(best_set)
    current_size = best_size
    
    for sa_restart in range(1000):
        if time.time() > deadline - 1:
            break
        
        # Start from current best
        current_set = set(current_best)
        
        # Perform several swap moves
        for step in range(50):
            if time.time() > deadline - 1:
                break
            
            # Try a random swap: remove one element, add one or more
            if not current_set:
                break
                
            elem_to_remove = random.choice(list(current_set))
            current_set.remove(elem_to_remove)
            
            # Try to add random elements
            candidates = [j for j in range(1, N + 1) if j not in current_set]
            if not candidates:
                current_set.add(elem_to_remove)
                continue
                
            added_count = 0
            for c in random.sample(candidates, min(len(candidates), 20)):
                if can_add_element(c, current_set):
                    current_set.add(c)
                    added_count += 1
            
            # Accept if better or equal (with some probability for worse)
            if len(current_set) >= current_size:
                if len(current_set) > current_size:
                    current_size = len(current_set)
                    current_best = current_set
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set
            else:
                # Reject: restore
                current_set.add(elem_to_remove)
                for c in list(current_set):
                    if c != elem_to_remove and c not in best_set:
                        # This is getting complex, just restore to previous state
                        pass
                # Simpler: just reject
                current_set = set(current_best)
    
    return sorted(best_set)
