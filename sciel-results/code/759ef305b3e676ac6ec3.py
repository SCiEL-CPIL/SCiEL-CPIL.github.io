import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    N = 256
    weights = list(range(1, N + 1))
    
    # Build conflict graph
    conflict = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if math.gcd(i + 1, j + 1) > 1:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Precompute sorted order by weight descending for faster bound
    sorted_indices = sorted(range(N), key=lambda i: weights[i], reverse=True)
    
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

    # Helper: get index that maximizes (weight - conflict_count_in_candidates)
    # This picks vertices that are both heavy and have many conflicts (good pruning)
    def get_branch_idx(mask):
        best_idx = -1
        best_score = -1
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            # Score: weight + number of conflicts with other candidates
            conflict_count = bin(conflict[idx] & mask).count('1')
            score = weights[idx] + conflict_count
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
        # If v is in candidates and (conflict[v] & candidates) == 0, we must include v.
        changed = True
        while changed:
            changed = False
            # Iterate through candidates
            c = candidates
            while c:
                lsb = c & (-c)
                idx = lsb.bit_length() - 1
                # Check if idx has any conflicts with other candidates
                # conflict[idx] contains neighbors. If (conflict[idx] & candidates) == 0, it's independent.
                if (conflict[idx] & candidates) == 0:
                    # Include idx
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
        
        # Upper bound: sum of all remaining candidate weights
        bound = current_weight + sum_weights(candidates)
        if bound <= best_weight[0]:
            return
        
        # Select vertex to branch on: maximize weight + conflict_count
        v = get_branch_idx(candidates)
        if v == -1:
            return
        
        # Branch 1: Include v
        # New candidates: candidates excluding v and its neighbors
        new_candidates = candidates & ~(conflict[v] | (1 << v))
        branch_and_bound(new_candidates, current_weight + weights[v], current_set | (1 << v))
        
        # Branch 2: Exclude v
        new_candidates = candidates & ~(1 << v)
        branch_and_bound(new_candidates, current_weight, current_set)

    # Greedy initial solution
    def greedy_initial():
        s = 0
        remaining = (1 << N) - 1
        # Sort by weight descending
        order = sorted(range(N), key=lambda i: weights[i], reverse=True)
        for idx in order:
            if (remaining & (1 << idx)) == 0:
                continue
            # Check compatibility with s
            if (conflict[idx] & s) == 0:
                s |= (1 << idx)
                remaining &= ~(1 << idx)
        return s, sum_weights(s)

    # Local search: try adding/removing/swapping to improve
    def local_search(initial_set):
        nonlocal best_weight, best_set
        current_set = initial_set
        current_weight = sum_weights(current_set)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding each excluded element
            excluded = ((1 << N) - 1) & ~current_set
            e = excluded
            while e:
                lsb = e & (-e)
                idx = lsb.bit_length() - 1
                if (conflict[idx] & current_set) == 0:
                    new_set = current_set | (1 << idx)
                    new_weight = current_weight + weights[idx]
                    if new_weight > current_weight:
                        current_set = new_set
                        current_weight = new_weight
                        improved = True
                e ^= lsb
            
            # Try removing one element and adding two that were blocked by it
            c = current_set
            while c and time.time() < deadline:
                lsb = c & (-c)
                idx = lsb.bit_length() - 1
                # What was blocked by idx?
                blocked = conflict[idx] & ~current_set
                if blocked:
                    # Try removing idx and adding best from blocked
                    b = blocked
                    best_add = -1
                    best_add_w = -1
                    while b:
                        blsb = b & (-b)
                        bidx = blsb.bit_length() - 1
                        # Check if bidx is compatible with current_set minus idx
                        if (conflict[bidx] & (current_set & ~(1 << idx))) == 0:
                            if weights[bidx] > best_add_w:
                                best_add_w = weights[bidx]
                                best_add = bidx
                        b ^= blsb
                    
                    if best_add >= 0:
                        new_set = (current_set & ~(1 << idx)) | (1 << best_add)
                        new_weight = current_weight - weights[idx] + weights[best_add]
                        if new_weight > current_weight:
                            current_set = new_set
                            current_weight = new_weight
                            improved = True
                c ^= lsb
        
        if current_weight > best_weight[0]:
            best_weight[0] = current_weight
            best_set[0] = current_set
        return current_set, current_weight

    init_set, init_weight = greedy_initial()
    best_weight[0] = init_weight
    best_set[0] = init_set
    
    all_candidates = (1 << N) - 1
    branch_and_bound(all_candidates, 0, 0)
    
    # Polish with local search
    if time.time() < deadline:
        local_search(best_set[0])
    
    # Convert bitmask to list of ints
    result = []
    mask = best_set[0]
    while mask:
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        mask ^= lsb
    
    return result
