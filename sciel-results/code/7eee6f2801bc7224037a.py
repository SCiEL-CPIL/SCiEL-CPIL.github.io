import time
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 58.0

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    def greedy_init():
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def local_improve(mask):
        improved = True
        while improved:
            improved = False
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    def modular_residue_init():
        # Use modular arithmetic to seed a good initial solution
        # Squares mod small primes have restricted residues
        # Try partitioning by residue classes modulo various moduli
        
        best_mask = 0
        best_size = 0
        
        # Try several moduli and residue-based initializations
        for mod in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            # Group candidates by residue mod `mod`
            residue_groups = {}
            m = cand_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                r = (v + 1) % mod
                if r not in residue_groups:
                    residue_groups[r] = 0
                residue_groups[r] |= lsb
                m ^= lsb
            
            # Try each residue class as starting set
            for r, group_mask in residue_groups.items():
                # Expand greedily from this seed
                mask = group_mask
                # Add vertices that don't conflict
                available = cand_mask & ~mask
                added = True
                while added:
                    added = False
                    m2 = available
                    while m2:
                        lsb2 = m2 & -m2
                        v2 = lsb2.bit_length() - 1
                        m2 ^= lsb2
                        if (adj[v2] & mask) == 0:
                            mask |= lsb2
                            available &= ~lsb2
                            added = True
                            break
                
                size = mask.bit_count()
                if size > best_size:
                    best_size = size
                    best_mask = mask
        
        return best_mask, best_size

    def swap_local_search(mask, max_swaps=5000):
        # 1-2 exchange local search: try swapping one vertex out for another
        # or removing one to add two
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps and time.time() < DEADLINE:
            improved = False
            swap_count += 1
            
            # Try single additions
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-2 swaps: remove one vertex, add two others
            in_mask = mask
            m_in = in_mask
            while m_in:
                lsb_in = m_in & -m_in
                v_out = lsb_in.bit_length() - 1
                m_in ^= lsb_in
                
                # Try removing v_out and adding two vertices that were blocked by it
                mask_without = mask & ~lsb_in
                # Find vertices that are now addable
                newly_addable = cand_mask & ~mask_without
                m_new = newly_addable
                add_list = []
                while m_new and len(add_list) < 2:
                    lsb_new = m_new & -m_new
                    v_new = lsb_new.bit_length() - 1
                    m_new ^= lsb_new
                    if (adj[v_new] & mask_without) == 0:
                        add_list.append(v_new)
                
                if len(add_list) >= 2:
                    # Check if adding both is valid
                    v1, v2 = add_list[0], add_list[1]
                    if (adj[v1] & mask_without) == 0 and (adj[v2] & mask_without) == 0:
                        # Check v1 and v2 don't conflict with each other
                        if (adj[v1] & (1 << v2)) == 0:
                            new_mask = mask_without | (1 << v1) | (1 << v2)
                            if new_mask.bit_count() > mask.bit_count():
                                mask = new_mask
                                improved = True
                                break
                elif len(add_list) == 1:
                    # 1-for-1 swap: remove v_out, add v_new (no net gain but might enable future)
                    # Skip for now, focus on size-increasing moves
                    pass
        
        return mask

    def multi_start_local_search(num_starts=50):
        best_mask = 0
        best_size = 0
        
        for s in range(num_starts):
            if time.time() > DEADLINE - 2.0:
                break
            
            # Different initialization strategies
            if s == 0:
                mask = greedy_init()
            elif s == 1:
                mask, _ = modular_residue_init()
            else:
                # Random perturbation of best found so far
                if best_mask == 0:
                    mask = greedy_init()
                else:
                    mask = best_mask
                    # Randomly remove k vertices
                    k = random.randint(1, 5)
                    m = mask
                    for _ in range(k):
                        if m == 0:
                            break
                        # Pick random bit
                        bits = []
                        mm = m
                        while mm:
                            lsb = mm & -mm
                            bits.append(lsb)
                            mm ^= lsb
                        if bits:
                            mask &= ~random.choice(bits)
                            m = mask
                    # Re-expand
                    mask = local_improve(mask)
            
            # Apply swap local search
            mask = swap_local_search(mask)
            mask = local_improve(mask)
            
            size = mask.bit_count()
            if size > best_size:
                best_size = size
                best_mask = mask
        
        return best_mask, best_size

    # Main: Use multi-start local search with modular residue seeding
    best_size = 0
    best_set = 0
    
    # Quick greedy for initial bound
    mask = greedy_init()
    mask = local_improve(mask)
    best_size = mask.bit_count()
    best_set = mask
    
    # Modular residue initialization
    mod_mask, mod_size = modular_residue_init()
    if mod_size > best_size:
        best_size = mod_size
        best_set = mod_mask
    
    # Multi-start local search with remaining time
    remaining_time = DEADLINE - time.time()
    num_starts = max(10, int(remaining_time * 5))  # Scale starts with available time
    best_set, best_size = multi_start_local_search(num_starts)
    
    # Final polish
    if time.time() < DEADLINE - 1.0:
        final_mask = local_improve(best_set)
        if final_mask.bit_count() > best_size:
            best_size = final_mask.bit_count()
            best_set = final_mask
    
    # Convert best_set to list of values (1-based)
    result = []
    m = best_set
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
