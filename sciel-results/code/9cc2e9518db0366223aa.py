import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 256
    
    # Perfect squares up to 2*256 = 512
    squares = set()
    i = 1
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    
    if n == 0:
        return [1]
    
    # Build conflict graph using bitsets
    # conflicts[i] is a bitmask of valid node indices that conflict with valid_nodes[i]
    conflicts = [0] * n
    for i in range(n):
        v = valid_nodes[i]
        for j in range(i + 1, n):
            u = valid_nodes[j]
            if v + u in squares:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    def is_valid_mask(mask):
        """Verify that the mask represents an independent set."""
        for i in range(n):
            if mask & (1 << i):
                if conflicts[i] & mask:
                    return False
        return True
    
    def greedy_mis(order):
        """Greedy: process nodes in given order, add if no conflict with already selected."""
        chosen_mask = 0
        for idx in order:
            if (conflicts[idx] & chosen_mask) == 0:
                chosen_mask |= (1 << idx)
        return chosen_mask
    
    def local_search(mask, max_iters=5000):
        """
        Improve the independent set by:
        1. Adding any node that doesn't conflict
        2. Swapping: remove one node, add another if it improves size or diversifies
        3. 2-for-1 swaps: remove two nodes, add one if it improves size
        """
        mask = mask
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and (time.time() - start_time) < deadline - 1:
            improved = False
            iter_count += 1
            
            # Phase 1: Try adding any single node that doesn't conflict
            for i in range(n):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
            
            if improved:
                continue
            
            # Phase 2: Try 1-for-1 swaps (remove one, add one)
            # This helps escape local optima where no single add is possible
            for i in range(n):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    mask |= (1 << i)
                    improved = True
                    break
                # Try removing each conflicting node
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    new_mask = (mask & ~r_bit) | (1 << i)
                    if (conflicts[i] & (mask & ~r_bit)) == 0:
                        mask = new_mask
                        improved = True
                        break
                    temp ^= r_bit
                if improved:
                    break
            
            # Phase 3: Try 2-for-1 swaps (remove two, add one) - less frequent
            if not improved and iter_count % 5 == 0:
                for i in range(n):
                    if mask & (1 << i):
                        continue
                    conf_in_mask = conflicts[i] & mask
                    if conf_in_mask == 0:
                        mask |= (1 << i)
                        improved = True
                        break
                    # Get list of conflicting indices
                    conf_list = []
                    temp = conf_in_mask
                    while temp:
                        r_bit = temp & (-temp)
                        r_idx = r_bit.bit_length() - 1
                        conf_list.append(r_idx)
                        temp ^= r_bit
                    
                    # Try removing pairs of conflicting nodes
                    for a in range(len(conf_list)):
                        for b in range(a + 1, len(conf_list)):
                            a_bit = 1 << conf_list[a]
                            b_bit = 1 << conf_list[b]
                            new_mask = mask & ~a_bit & ~b_bit
                            if (conflicts[i] & new_mask) == 0:
                                mask = new_mask | (1 << i)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return mask
    
    def min_degree_greedy():
        """Min-degree greedy with bitmask operations."""
        current_mask = (1 << n) - 1
        sol_mask = 0
        
        while current_mask:
            min_deg = n + 1
            best_candidates = []
            
            for v in range(n):
                if current_mask & (1 << v):
                    deg = (conflicts[v] & current_mask).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_candidates = [v]
                    elif deg == min_deg:
                        best_candidates.append(v)
            
            if not best_candidates:
                break
            
            # Randomly pick from min-degree candidates
            v = random.choice(best_candidates)
            sol_mask |= (1 << v)
            current_mask &= ~(conflicts[v] | (1 << v))
        
        return sol_mask
    
    best_mask = 0
    best_size = 0
    
    # Mod 3 baseline: v % 3 == 1
    # a,b ≡ 1 (mod 3) => a+b ≡ 2 (mod 3), not a square
    # 2a ≡ 2 (mod 3), not a square
    mod3_mask = 0
    for v in valid_nodes:
        if v % 3 == 1:
            mod3_mask |= (1 << valid_nodes.index(v))
    best_mask = mod3_mask
    best_size = bin(mod3_mask).count('1')
    
    random.seed(42)
    
    # Multiple restarts with min-degree greedy + local search
    num_restarts = 100
    for r in range(num_restarts):
        if (time.time() - start_time) > deadline - 2:
            break
        
        # Min-degree greedy
        mask = min_degree_greedy()
        mask = local_search(mask, max_iters=2000)
        
        if bin(mask).count('1') > best_size:
            best_size = bin(mask).count('1')
            best_mask = mask
    
    # Additional random-order greedy restarts
    for r in range(50):
        if (time.time() - start_time) > deadline - 2:
            break
        
        order = list(range(n))
        random.shuffle(order)
        mask = greedy_mis(order)
        mask = local_search(mask, max_iters=2000)
        
        if bin(mask).count('1') > best_size:
            best_size = bin(mask).count('1')
            best_mask = mask
    
    # Final aggressive local search on best
    if (time.time() - start_time) < deadline - 1:
        best_mask = local_search(best_mask, max_iters=5000)
        best_size = bin(best_mask).count('1')
    
    # Convert mask back to values
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(valid_nodes[i])
    
    result = sorted(result)
    
    if not result:
        result = [1]
    
    return result
