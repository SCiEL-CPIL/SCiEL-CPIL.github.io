import math
import time

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
    
    # Precompute smallest prime factor for upper bound
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N+1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For upper bound: partition by smallest prime factor
    # For each prime p, at most one number with spf=p can be in solution
    # Number 1 has no spf constraint
    spf_groups = {}  # spf -> list of indices
    for i in range(N):
        val = i + 1
        if val == 1:
            spf_groups.setdefault(0, []).append(i)
        else:
            spf_groups.setdefault(spf[val], []).append(i)
    
    # Precompute for each prime group the max weight
    # But we need dynamic: for a given candidate mask, for each spf group, max weight among candidates in that group
    
    best_weight = [0]
    best_set = [0]
    
    # Helper: sum weights of bits in mask
    def sum_weights(mask):
        s = 0
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            s += weights[idx]
            mask ^= lsb
        return s

    # Tighter upper bound using SPF partition
    def tight_bound(candidates):
        # For each spf group, take max weight among candidates in that group
        total = 0
        for sp, idxs in spf_groups.items():
            max_w = 0
            for idx in idxs:
                if (candidates >> idx) & 1:
                    if weights[idx] > max_w:
                        max_w = weights[idx]
            total += max_w
        return total

    # Helper: get index to branch on - use degree-weighted heuristic
    def get_branch_idx(mask):
        best_idx = -1
        best_score = -1
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            # Score: weight * (1 + degree_in_candidates)
            deg = bin(conflict[idx] & mask).count('1')
            score = weights[idx] * (1 + deg)
            if score > best_score:
                best_score = score
                best_idx = idx
            mask ^= lsb
        return best_idx

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
        
        # Upper bound using SPF partition
        bound = current_weight + tight_bound(candidates)
        if bound <= best_weight[0]:
            return
        
        # Select vertex to branch on
        v = get_branch_idx(candidates)
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
    
    result = []
    mask = best_set[0]
    while mask:
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        mask ^= lsb
    
    return result
