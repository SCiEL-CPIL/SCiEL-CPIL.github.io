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
        Improves the independent set using 1-0, 1-1, 1-2, and 2-1 moves.
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
            m_u = mask
            found_move = False
            while m_u and not found_move:
                lsb_u = m_u & -m_u
                u = lsb_u.bit_length() - 1
                m_u ^= lsb_u
                
                mask_without_u = mask ^ lsb_u
                
                # Find candidates v not in mask such that adj[v] & mask_without_u == 0
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
                    
                n_valid = len(valid_v_list)
                for i in range(n_valid):
                    v = valid_v_list[i]
                    for j in range(i + 1, n_valid):
                        w = valid_v_list[j]
                        if (adj[v] & (1 << w)) == 0:
                            mask = mask_without_u | (1 << v) | (1 << w)
                            improved = True
                            found_move = True
                            break
                    if found_move:
                        break
                            
            if found_move:
                continue

            # 3. Try 2-1 moves: Remove u, v (both in mask), add w (not in mask)
            # This keeps size same but may enable future improvements
            # Actually, 2-1 reduces size by 1, so only useful as part of a sequence
            # Let's try 1-1 swaps that might lead to better configurations
            # Try 1-1: remove u from mask, add v not in mask, such that the new set is valid
            # This doesn't change size but can help escape local optima
            m_u = mask
            found_swap = False
            while m_u and not found_swap:
                lsb_u = m_u & -m_u
                u = lsb_u.bit_length() - 1
                m_u ^= lsb_u
                
                mask_without_u = mask ^ lsb_u
                
                # Find v not in mask such that adj[v] & mask_without_u == 0
                # AND v is NOT adjacent to u (otherwise we could have added v directly)
                # Actually we just need v compatible with mask_without_u
                m_v = cand_mask & ~mask
                while m_v:
                    lsb_v = m_v & -m_v
                    v = lsb_v.bit_length() - 1
                    m_v ^= lsb_v
                    if (adj[v] & mask_without_u) == 0:
                        # Check if this swap actually helps: is v connected to more 
                        # non-mask nodes than u was? Or just accept it for diversification
                        # Only do the swap if it reduces the "blocking" of other nodes
                        # Simple heuristic: do it if v has fewer connections to non-mask
                        # than u had
                        u_block = (adj[u] & cand_mask & ~mask).bit_count()
                        v_block = (adj[v] & cand_mask & ~mask).bit_count()
                        if v_block < u_block:
                            mask = mask_without_u | lsb_v
                            improved = True
                            found_swap = True
                            break
                            
            if found_swap:
                continue

            # 4. Try 2-1 moves: remove u,v from mask, add w
            # Size decreases by 1, but may unblock a 1-2 move later
            # This is expensive, so only try a limited number
            m_u = mask
            found_21 = False
            count_21_tried = 0
            while m_u and not found_21 and count_21_tried < 20:
                lsb_u = m_u & -m_u
                u = lsb_u.bit_length() - 1
                m_u ^= lsb_u
                
                m_v = mask & ~lsb_u
                while m_v and not found_21 and count_21_tried < 20:
                    lsb_v = m_v & -m_v
                    v = lsb_v.bit_length() - 1
                    m_v ^= lsb_v
                    count_21_tried += 1
                    
                    mask_without_uv = mask ^ lsb_u ^ lsb_v
                    
                    # Find w not in mask such that adj[w] & mask_without_uv == 0
                    # AND w is adjacent to u or v (otherwise 1-0 would have worked)
                    m_w = cand_mask & ~mask
                    while m_w:
                        lsb_w = m_w & -m_w
                        w = lsb_w.bit_length() - 1
                        m_w ^= lsb_w
                        if (adj[w] & mask_without_uv) == 0:
                            # Check w is blocked by u or v
                            if (adj[w] & lsb_u) or (adj[w] & lsb_v):
                                mask = mask_without_uv | lsb_w
                                improved = True
                                found_21 = True
                                break
                                
            if found_21:
                continue

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
    
    # Try multiple perturbations with different strengths
    while time.time() - start_time < DEADLINE:
        # Perturb: remove 1-3 vertices
        k_remove = random.randint(1, 3)
        current_mask = perturb(current_mask, k_remove)
        
        # Local Search
        current_mask = local_search(current_mask)
        
        if current_mask.bit_count() > best_size:
            best_size = current_mask.bit_count()
            best_mask = current_mask

        # Occasionally, try a fresh greedy to explore different basins
        if random.random() < 0.05:
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
