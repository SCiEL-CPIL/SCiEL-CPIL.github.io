import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 512
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # d can range from 1 to (n-1)//3
    # a can range from 1 to n - 3d
    all_aps = []
    # Map each number to the indices of APs it is part of
    ap_indices_by_num = {i: [] for i in range(1, n + 1)}
    
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for x in ap:
                ap_indices_by_num[x].append(idx)
                
    num_aps = len(all_aps)
    
    def can_add(x, S):
        """
        Check if x can be added to set S without creating a 4-AP.
        S is a set of integers.
        """
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            # Check if the other 3 elements are in S
            # Since x is in ap, we check the other 3
            count_in_S = 0
            for elem in ap:
                if elem != x and elem in S:
                    count_in_S += 1
            if count_in_S == 3:
                return False
        return True

    def greedy_fill(S, order):
        """
        Greedily add elements from 'order' to set S if they don't create a 4-AP.
        Returns the new set.
        """
        S = set(S)
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S

    # Strategy 1: Initial Construction using multiple heuristics
    best_set = set()
    
    # Heuristic 1: Greedy from 1 to n
    S1 = greedy_fill(set(), range(1, n + 1))
    
    # Heuristic 2: Greedy from n to 1
    S2 = greedy_fill(set(), range(n, 0, -1))
    
    # Heuristic 3: Base-4 {0,1} digits (Salem-Spencer style, 3-AP free hence 4-AP free)
    S3 = set()
    for i in range(1, n + 1):
        valid = True
        temp = i
        while temp > 0:
            if temp % 4 > 1:
                valid = False
                break
            temp //= 4
        if valid:
            S3.add(i)
    # Try to extend S3
    S3 = greedy_fill(S3, range(1, n + 1))
    
    # Heuristic 4: Randomized Greedy
    S4 = set()
    rng = random.Random(42)
    order = list(range(1, n + 1))
    rng.shuffle(order)
    S4 = greedy_fill(set(), order)
    
    candidates = [S1, S2, S3, S4]
    best_set = max(candidates, key=len)
    
    # Strategy 2: Local Search with 1-exchange and 2-exchange (Repair)
    # We will iterate, trying to improve the best set found.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_best = set(best_set)
        
        # Phase 2a: Simple Additions
        # Try to add any element that can be added
        for x in range(1, n + 1):
            if x not in current_best and can_add(x, current_best):
                current_best.add(x)
                improved = True
        
        best_set = set(current_best)
        
        # Phase 2b: 1-Exchange (Remove 1, Add more)
        # Try removing each element and see if we can add more than 1 back
        elements_to_try = list(best_set)
        random.shuffle(elements_to_try)
        
        for x in elements_to_try:
            if time.time() > deadline:
                break
                
            candidate = set(best_set)
            candidate.remove(x)
            
            # Greedily refill
            # Order: try all numbers not in candidate
            refill_order = [y for y in range(1, n + 1) if y not in candidate]
            random.shuffle(refill_order)
            
            new_candidate = greedy_fill(candidate, refill_order)
            
            if len(new_candidate) > len(best_set):
                best_set = new_candidate
                improved = True
                break # Restart the loop with the new best
                
    # Strategy 3: More aggressive Local Search with Perturbation and Simulated Annealing-like moves
    # We try to escape local optima by removing a few elements and refilling.
    
    for restart in range(500):
        if time.time() > deadline - 2:
            break
            
        current_set = set(best_set)
        
        # Perturb: remove 1 to 3 random elements
        num_remove = random.randint(1, 3)
        to_remove = random.sample(list(current_set), min(num_remove, len(current_set)))
        for x in to_remove:
            current_set.remove(x)
            
        # Refill greedily with random order
        refill_order = [y for y in range(1, n + 1) if y not in current_set]
        random.shuffle(refill_order)
        
        new_set = greedy_fill(current_set, refill_order)
        
        if len(new_set) > len(best_set):
            best_set = new_set
            
    # Final Polish: Try to add any remaining elements
    for x in range(1, n + 1):
        if x not in best_set and can_add(x, best_set):
            best_set.add(x)
            
    return sorted(best_set)

# Test/Verification logic (not part of the output, but good for local dev)
# def verify(S):
#     S_set = set(S)
#     for ap in all_aps:
#         if all(x in S_set for x in ap):
#             return False
#     return True
