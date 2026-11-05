import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency bitmasks
    # adj[i] contains bits j such that (i+1) + (j+1) is a square
    # Note: i and j are 0-indexed. Value is i+1.
    adj = [0] * N
    forbidden = [False] * N
    
    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
            continue
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
        
    # Precompute degrees for greedy
    degrees = [0] * N
    for i in range(N):
        if not forbidden[i]:
            degrees[i] = adj[i].bit_count()

    def greedy_init():
        available_mask = cand_mask
        selected_mask = 0
        # To speed up, we can pick min degree dynamically
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                # Dynamic degree in available subgraph
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            # Remove best_v and its neighbors from available
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def local_search(mask):
        """
        Improves the independent set using 1-1 and 1-2 moves.
        """
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE:
                break
                
            # 1. Try to add a vertex (1-0 move)
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

            # 2. Try 1-2 moves: Remove u, add v, w
            # This increases size by 1.
            # We iterate u in mask.
            m_u = mask
            found_move = False
            while m_u and not found_move:
                lsb_u = m_u & -m_u
                u = lsb_u.bit_length() - 1
                m_u ^= lsb_u
                
                # Mask of nodes in mask except u
                mask_without_u = mask ^ lsb_u
                
                # Find candidates v not in mask such that adj[v] & mask_without_u == 0
                # These are nodes that are compatible with mask \ {u}
                # Let's collect them
                valid_v_list = []
                m_v = cand_mask & ~mask
                while m_v:
                    lsb_v = m_v & -m_v
                    v = lsb_v.bit_length() - 1
                    m_v ^= lsb_v
                    if (adj[v] & mask_without_u) == 0:
                        valid_v_list.append(v)
                        
                if len(valid_v_list) < 2:
                    continue
                    
                # Now find a pair v, w in valid_v_list that are not adjacent
                # valid_v_list size is small usually?
                # Worst case, many are compatible.
                # We just need one pair.
                n_valid = len(valid_v_list)
                for i in range(n_valid):
                    v = valid_v_list[i]
                    # Check against all j > i
                    for j in range(i + 1, n_valid):
                        w = valid_v_list[j]
                        if (adj[v] & (1 << w)) == 0:
                            # Move is valid
                            mask = mask_without_u | (1 << v) | (1 << w)
                            improved = True
                            found_move = True
                            break
                    if found_move:
                        break
                            
            if not found_move:
                # Try 1-1 swaps? (Size constant, might help escape)
                # Or just stop. 1-2 is stronger for maximizing size.
                # If we can't add, we might be stuck.
                # Let's try a simple 1-1 swap that allows a subsequent add?
                # That's complex. Let's rely on perturbation in ILS.
                pass
                
        return mask

    def perturb(mask, k):
        """Remove k random vertices"""
        m = mask
        for _ in range(k):
            if not m:
                break
            count = m.bit_count()
            r = random.randint(1, count)
            cnt = 0
            temp = m
            while temp:
                lsb = temp & -temp
                cnt += 1
                if cnt == r:
                    m ^= lsb
                    break
                temp ^= lsb
        return m

    # Main ILS Loop
    best_mask = greedy_init()
    best_mask = local_search(best_mask)
    best_size = best_mask.bit_count()
    
    current_mask = best_mask
    
    while time.time() - start_time < DEADLINE:
        # Perturb
        # Remove 2-4 vertices
        k_remove = random.randint(2, 4)
        current_mask = perturb(current_mask, k_remove)
        
        # Local Search
        current_mask = local_search(current_mask)
        
        if current_mask.bit_count() > best_size:
            best_size = current_mask.bit_count()
            best_mask = current_mask
            # Restart from best? Or continue from current?
            # Usually ILS continues from current, but if we found a new best,
            # it's good to keep exploring around it.
            
        # Occasionally, try a fresh greedy to explore different basins
        if random.random() < 0.1:
            fresh = greedy_init()
            fresh = local_search(fresh)
            if fresh.bit_count() > best_size:
                best_size = fresh.bit_count()
                best_mask = fresh
                current_mask = fresh

    # Convert to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
