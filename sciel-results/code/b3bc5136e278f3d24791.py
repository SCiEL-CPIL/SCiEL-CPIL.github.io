import time
import random

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 55
    
    N = 256
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    def popcount(x):
        return bin(x).count('1')
    
    def bits(x):
        res = []
        while x:
            lsb = x & -x
            res.append(lsb.bit_length() - 1)
            x ^= lsb
        return res

    all_mask = (1 << (N + 1)) - 2
    
    def greedy_construct(strategy='min'):
        """Greedy with different strategies: min-degree, max-degree, random"""
        selected_mask = 0
        remaining_mask = all_mask
        while remaining_mask:
            r = remaining_mask
            best_node = -1
            best_deg = -1 if strategy == 'max' else N + 1
            best_nodes = []
            
            if strategy == 'random':
                # Pick random node
                bit_list = bits(remaining_mask)
                best_node = random.choice(bit_list)
            else:
                while r:
                    lsb = r & -r
                    node = lsb.bit_length() - 1
                    deg = popcount(conflict_mask[node] & remaining_mask)
                    if strategy == 'min':
                        if deg < best_deg:
                            best_deg = deg
                            best_node = node
                            best_nodes = [node]
                        elif deg == best_deg:
                            best_nodes.append(node)
                    else:  # max
                        if deg > best_deg:
                            best_deg = deg
                            best_node = node
                            best_nodes = [node]
                        elif deg == best_deg:
                            best_nodes.append(node)
                    r ^= lsb
                
                if best_nodes:
                    best_node = random.choice(best_nodes)
            
            if best_node == -1:
                break
                
            selected_mask |= (1 << best_node)
            remaining_mask &= ~(1 << best_node)
            remaining_mask &= ~conflict_mask[best_node]
            
        return selected_mask

    def local_search(start_mask):
        cur_mask = start_mask
        cur_size = popcount(cur_mask)
        removed_mask = all_mask & ~cur_mask
        
        # 1. Add back non-conflicting
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            r = removed_mask
            while r:
                lsb = r & -r
                node = lsb.bit_length() - 1
                if (conflict_mask[node] & cur_mask) == 0:
                    cur_mask |= lsb
                    removed_mask &= ~lsb
                    cur_size += 1
                    improved = True
                r ^= lsb
        
        # 2. Try 1-for-1 swaps and 2-for-2 swaps
        for _ in range(15):
            if time.time() > deadline - 2:
                break
            
            if removed_mask == 0:
                break
            r_bits = bits(removed_mask)
            if not r_bits:
                break
            
            # Try 1-for-1 swap
            r_node = random.choice(r_bits)
            r_bit = 1 << r_node
            conflicts = conflict_mask[r_node] & cur_mask
            
            if conflicts == 0:
                cur_mask |= r_bit
                removed_mask &= ~r_bit
                cur_size += 1
                # Add back loop
                improved = True
                while improved and time.time() < deadline - 2:
                    improved = False
                    rem = removed_mask
                    while rem:
                        lsb = rem & -rem
                        node = lsb.bit_length() - 1
                        if (conflict_mask[node] & cur_mask) == 0:
                            cur_mask |= lsb
                            removed_mask &= ~lsb
                            cur_size += 1
                            improved = True
                        rem ^= lsb
                continue
            
            if popcount(conflicts) == 1:
                s_bit = conflicts
                if (conflict_mask[r_node] & (cur_mask ^ s_bit)) == 0:
                    cur_mask ^= s_bit
                    cur_mask |= r_bit
                    removed_mask ^= s_bit
                    removed_mask ^= r_bit
                    # Add back loop
                    improved = True
                    while improved and time.time() < deadline - 2:
                        improved = False
                        rem = removed_mask
                        while rem:
                            lsb = rem & -rem
                            node = lsb.bit_length() - 1
                            if (conflict_mask[node] & cur_mask) == 0:
                                cur_mask |= lsb
                                removed_mask &= ~lsb
                                cur_size += 1
                                improved = True
                            rem ^= lsb
            
            # Try 2-for-2 swap: pick two removed nodes, see if we can swap with two selected
            elif random.random() < 0.3:
                if len(r_bits) >= 2:
                    r1, r2 = random.sample(r_bits, 2)
                    r1_bit, r2_bit = 1 << r1, 1 << r2
                    c1 = conflict_mask[r1] & cur_mask
                    c2 = conflict_mask[r2] & cur_mask
                    
                    # Check if r1 and r2 conflict with each other
                    if (conflict_mask[r1] & r2_bit) or (conflict_mask[r2] & r1_bit):
                        continue
                    
                    # Get the set of conflicts
                    conflict_set = c1 | c2
                    if popcount(conflict_set) <= 2:
                        # Check if we can swap out the conflicts
                        s_mask = conflict_set
                        # Check r1 and r2 don't conflict with cur_mask minus s_mask
                        if (conflict_mask[r1] & (cur_mask ^ s_mask)) == 0 and \
                           (conflict_mask[r2] & (cur_mask ^ s_mask)) == 0:
                            # Perform swap
                            cur_mask ^= s_mask
                            cur_mask |= r1_bit
                            cur_mask |= r2_bit
                            removed_mask ^= s_mask
                            removed_mask ^= r1_bit
                            removed_mask ^= r2_bit
                            # cur_size unchanged
                            # Add back loop
                            improved = True
                            while improved and time.time() < deadline - 2:
                                improved = False
                                rem = removed_mask
                                while rem:
                                    lsb = rem & -rem
                                    node = lsb.bit_length() - 1
                                    if (conflict_mask[node] & cur_mask) == 0:
                                        cur_mask |= lsb
                                        removed_mask &= ~lsb
                                        cur_size += 1
                                        improved = True
                                    rem ^= lsb
        
        return cur_mask, cur_size

    # Try multiple greedy strategies
    best_mask = None
    best_size = 0
    strategies = ['min', 'max', 'random', 'random', 'min', 'max']
    for strat in strategies:
        if time.time() > deadline - 10:
            break
        m = greedy_construct(strat)
        s = popcount(m)
        if s > best_size:
            best_size = s
            best_mask = m
    
    # Refine best
    best_mask, best_size = local_search(best_mask)
    
    # Random restarts with perturbation
    restart_count = 0
    while time.time() < deadline - 2:
        restart_count += 1
        
        # Perturb: remove random subset
        cur_mask = best_mask
        k = random.randint(1, 6)
        nodes = bits(cur_mask)
        if len(nodes) > k:
            to_remove = random.sample(nodes, k)
            for n in to_remove:
                cur_mask &= ~(1 << n)
        
        refined_mask, refined_size = local_search(cur_mask)
        
        if refined_size > best_size:
            best_size = refined_size
            best_mask = refined_mask
    
    return bits(best_mask)

# EVOLVE-BLOCK-END
