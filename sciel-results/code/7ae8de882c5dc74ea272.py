import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    weights = list(range(1, N + 1))
    
    # Build conflict graph
    conflict = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if math.gcd(i + 1, j + 1) > 1:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    best_weight = [0]
    best_set = [0]  # Use int bitmask for set
    
    # Helper: sum weights of bits in mask
    def sum_weights(mask):
        s = 0
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            s += weights[idx]
            mask ^= lsb
        return s

    # Helper: get index of highest weight bit in mask
    def get_max_weight_idx(mask):
        best_idx = -1
        best_w = -1
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            if weights[idx] > best_w:
                best_w = weights[idx]
                best_idx = idx
            mask ^= lsb
        return best_idx

    # Stronger upper bound: for each vertex, compute max weight of independent set
    # using a greedy approximation of the Lagrangian relaxation
    def lagrangian_bound(candidates, current_weight):
        """
        Upper bound: current_weight + sum over each candidate v of 
        max(0, weights[v] - sum of weights of its neighbors in candidates / 2)
        This is a valid upper bound for MWIS.
        """
        bound = current_weight
        c = candidates
        while c:
            lsb = c & (-c)
            idx = lsb.bit_length() - 1
            # Count how many of idx's neighbors are in candidates
            neighbors_in_cand = conflict[idx] & candidates
            # Upper bound contribution: weights[idx] - (sum of neighbor weights) / 2
            # But we need to be careful - use a simpler valid bound
            # Actually, a valid upper bound is: for each v, w(v) - (1/2) * sum_{u in N(v) ∩ C} w(u)
            # Sum this over all v in C, then take max with sum_weights
            c ^= lsb
        
        # Simpler valid upper bound: sum of weights of all candidates
        # But we can do better with a greedy matching-based bound
        # For now, use the simple bound and improve if needed
        return current_weight + sum_weights(candidates)

    def branch_and_bound(candidates, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
        
        # Reduction: Include all vertices that have no conflicts with any other candidate
        changed = True
        while changed:
            changed = False
            c = candidates
            while c:
                lsb = c & (-c)
                idx = lsb.bit_length() - 1
                if (conflict[idx] & candidates) == 0:
                    current_weight += weights[idx]
                    current_set |= (1 << idx)
                    candidates &= ~(1 << idx)
                    changed = True
                c ^= lsb
            
            if candidates == 0:
                if current_weight > best_weight[0]:
                    best_weight[0] = current_weight
                    best_set[0] = current_set
                return
        
        # Upper bound: sum of all candidate weights
        bound = current_weight + sum_weights(candidates)
        if bound <= best_weight[0]:
            return
        
        # Select vertex to branch on: highest weight
        v = get_max_weight_idx(candidates)
        if v == -1:
            return
        
        # Branch 1: Include v
        new_candidates = candidates & ~(conflict[v] | (1 << v))
        branch_and_bound(new_candidates, current_weight + weights[v], current_set | (1 << v))
        
        # Branch 2: Exclude v
        new_candidates = candidates & ~(1 << v)
        branch_and_bound(new_candidates, current_weight, current_set)

    # Greedy initial solution
    def greedy_initial():
        s = 0
        remaining = (1 << N) - 1
        order = sorted(range(N), key=lambda i: weights[i], reverse=True)
        for idx in order:
            if (remaining & (1 << idx)) == 0:
                continue
            if (conflict[idx] & s) == 0:
                s |= (1 << idx)
                remaining &= ~(1 << idx)
        return s, sum_weights(s)

    init_set, init_weight = greedy_initial()
    best_weight[0] = init_weight
    best_set[0] = init_set
    
    all_candidates = (1 << N) - 1
    branch_and_bound(all_candidates, 0, 0)
    
    # Convert bitmask to list of ints
    result = []
    mask = best_set[0]
    while mask:
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        mask ^= lsb
    
    return result
