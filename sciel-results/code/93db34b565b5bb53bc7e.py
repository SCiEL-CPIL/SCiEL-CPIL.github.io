import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    # Precompute reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
        
    # Precompute suffix sums of reciprocals for upper bound estimation
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recip[i]

    best_sum = 0.0
    best_set = []
    
    # Initial greedy solution to establish a baseline
    def get_greedy_solution():
        chosen = set()
        current_sum = 0.0
        for i in range(1, N + 1):
            conflict = False
            for a in chosen:
                b1 = 2 * i - a
                if b1 in chosen and b1 != i and b1 != a:
                    conflict = True
                    break
                b2 = 2 * a - i
                if b2 in chosen and b2 != i and b2 != a:
                    conflict = True
                    break
                if (i + a) % 2 == 0:
                    b3 = (i + a) // 2
                    if b3 in chosen and b3 != i and b3 != a:
                        conflict = True
                        break
            if not conflict:
                chosen.add(i)
                current_sum += recip[i]
        return current_sum, list(chosen)

    best_sum, best_set = get_greedy_solution()
    
    # Precompute for each v, the list of (w, u) pairs where adding v with w in chosen forbids u
    # Actually, let's precompute conflict_graph: for each pair (u,v), do they form part of a 3-AP?
    # Two elements u,v are "conflicting" if there exists w such that {u,v,w} is a 3-AP
    # This is equivalent to: u and v have a common third element in some 3-AP
    
    # For the B&B, when we add v, we need to forbid u if {u,v,w} is 3-AP for some w in chosen
    # Let's precompute: for each v, a dict mapping w -> set of u's forbidden
    
    # Actually, simpler: precompute for each v and w, the forbidden u's
    # forbidden_u[v][w] = list of u such that {u,v,w} is a 3-AP
    
    # Let me just use the bitmask approach with on-the-fly computation but optimized
    
    def compute_forbidden_mask(v, chosen_mask):
        """Compute bitmask of elements forbidden when adding v, given chosen_mask."""
        forbidden = 0
        cm = chosen_mask
        while cm:
            lowbit = cm & -cm
            w = lowbit.bit_length() - 1
            cm ^= lowbit
            
            # u1 = 2v - w
            u1 = 2 * v - w
            if 1 <= u1 <= N and u1 != v and u1 != w:
                forbidden |= (1 << u1)
            
            # u2 = 2w - v
            u2 = 2 * w - v
            if 1 <= u2 <= N and u2 != v and u2 != w:
                forbidden |= (1 << u2)
            
            # u3 = (v+w)/2
            if (v + w) % 2 == 0:
                u3 = (v + w) // 2
                if 1 <= u3 <= N and u3 != v and u3 != w:
                    forbidden |= (1 << u3)
        
        return forbidden
    
    # B&B with bitmask for chosen
    def dfs(available_mask, chosen_mask, chosen_sum, depth):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return False
        
        if available_mask == 0:
            if chosen_sum > best_sum:
                best_sum = chosen_sum
                s = []
                cm = chosen_mask
                while cm:
                    lowbit = cm & -cm
                    idx = lowbit.bit_length() - 1
                    s.append(idx)
                    cm ^= lowbit
                best_set = s
            return True
            
        # Find the smallest available index
        lowest_bit = available_mask & -available_mask
        idx = lowest_bit.bit_length() - 1
        
        # Upper bound using suffix sum
        if chosen_sum + suffix_sum[idx] <= best_sum:
            return True
            
        # Branch 1: Include idx
        forbidden = compute_forbidden_mask(idx, chosen_mask)
        new_available = available_mask & ~lowest_bit & ~forbidden
        new_chosen_mask = chosen_mask | lowest_bit
        
        dfs(new_available, new_chosen_mask, chosen_sum + recip[idx], depth + 1)
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude idx
        dfs(available_mask & ~lowest_bit, chosen_mask, chosen_sum, depth + 1)
        
        return True

    # Start DFS
    initial_mask = 0
    for i in range(1, N + 1):
        initial_mask |= (1 << i)
    
    dfs(initial_mask, 0, 0.0, 0)
    
    # Local search refinement
    if time.time() < deadline - 5:
        current_set = set(best_set)
        current_sum = sum(recip[i] for i in current_set)
        
        # Try adding elements
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(1, N + 1):
                if i in current_set:
                    continue
                can_add = True
                for a in current_set:
                    b1 = 2 * i - a
                    if b1 in current_set and b1 != i and b1 != a:
                        can_add = False
                        break
                    b2 = 2 * a - i
                    if b2 in current_set and b2 != i and b2 != a:
                        can_add = False
                        break
                    if (i + a) % 2 == 0:
                        b3 = (i + a) // 2
                        if b3 in current_set and b3 != i and b3 != a:
                            can_add = False
                            break
                if can_add:
                    current_set.add(i)
                    current_sum += recip[i]
                    improved = True
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = list(current_set)
        
        # Try swapping
        for remove_i in list(current_set):
            if time.time() > deadline - 2:
                break
            for add_i in range(1, N + 1):
                if add_i in current_set or add_i == remove_i:
                    continue
                temp_set = current_set - {remove_i}
                can_add = True
                for a in temp_set:
                    b1 = 2 * add_i - a
                    if b1 in temp_set and b1 != add_i and b1 != a:
                        can_add = False
                        break
                    b2 = 2 * a - add_i
                    if b2 in temp_set and b2 != add_i and b2 != a:
                        can_add = False
                        break
                    if (add_i + a) % 2 == 0:
                        b3 = (add_i + a) // 2
                        if b3 in temp_set and b3 != add_i and b3 != a:
                            can_add = False
                            break
                if can_add:
                    new_sum = current_sum - recip[remove_i] + recip[add_i]
                    if new_sum > best_sum:
                        current_set = temp_set | {add_i}
                        current_sum = new_sum
                        best_sum = current_sum
                        best_set = list(current_set)
    
    if not best_set:
        return [1]
        
    return best_set
