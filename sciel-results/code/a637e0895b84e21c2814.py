import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 255
    # 255 = 3 * 5 * 17
    
    # Build candidate list: all x in [1,255] that share a factor with 255
    # (since 255 must be in the set, every other element must have gcd > 1 with 255)
    def shares_factor_with_255(x):
        return math.gcd(x, 255) > 1
    
    candidates = [x for x in range(1, n + 1) if shares_factor_with_255(x)]
    m = len(candidates)
    cand_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = cand_idx[255]
    
    # Build adjacency as bitmasks
    adj = [0] * m
    for i in range(m):
        vi = candidates[i]
        for j in range(i + 1, m):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute degree within full graph
    deg = [adj[i].bit_count() for i in range(m)]
    
    # Multiple greedy strategies
    def greedy_clique(adj, m, start, deadline, strategy="max_deg"):
        clique_mask = 1 << start
        available = adj[start] & ~(1 << start)
        
        while available and time.time() < deadline:
            best_v = -1
            best_score = -1
            mask = available
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                deg_in_avail = (adj[v] & available).bit_count()
                if strategy == "max_deg":
                    score = deg_in_avail
                elif strategy == "max_deg_plus_orig":
                    score = deg_in_avail + deg[v]
                else:
                    score = deg_in_avail
                if score > best_score:
                    best_score = score
                    best_v = v
                mask ^= lb
            
            if best_v == -1:
                break
            clique_mask |= (1 << best_v)
            available &= adj[best_v]
            available &= ~(1 << best_v)
        
        return clique_mask
    
    best_size = 0
    best_mask = 0
    
    # Try multiple greedy strategies
    for strategy in ["max_deg", "max_deg_plus_orig"]:
        mask = greedy_clique(adj, m, idx_255, deadline, strategy)
        size = mask.bit_count()
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Also try starting from multiples of 3 as baseline
    multiples_3_mask = 0
    for x in range(3, 256, 3):
        multiples_3_mask |= (1 << cand_idx[x])
    if multiples_3_mask.bit_count() > best_size:
        best_size = multiples_3_mask.bit_count()
        best_mask = multiples_3_mask
    
    # Local search with swap moves
    if time.time() < deadline - 3:
        cur_mask = best_mask
        
        # Helper: check if adding v to clique (after removing u) is valid
        def can_swap(cur_mask, u, v, adj, m):
            """Check if removing u and adding v keeps it a clique."""
            if (cur_mask >> v) & 1:
                return False  # v already in clique
            if (cur_mask >> u) & 1 == 0:
                return False  # u not in clique
            clique_wo_u = cur_mask & ~(1 << u)
            # v must connect to all in clique_wo_u
            if (adj[v] & clique_wo_u) != clique_wo_u:
                return False
            return True
        
        improved = True
        max_rounds = 200
        round_num = 0
        while improved and time.time() < deadline - 3 and round_num < max_rounds:
            improved = False
            round_num += 1
            
            clique_list = [i for i in range(m) if (cur_mask >> i) & 1]
            not_in_clique = ((1 << m) - 1) & ~cur_mask
            
            for u in clique_list:
                if time.time() > deadline - 3:
                    break
                clique_wo_u = cur_mask & ~(1 << u)
                
                # Find best v to swap in
                best_v = -1
                best_score = -1
                mask = not_in_clique
                while mask:
                    lb = mask & (-mask)
                    v = lb.bit_length() - 1
                    if (adj[v] & clique_wo_u) == clique_wo_u:
                        # Valid swap. Score by how many in original clique it connects to
                        score = (adj[v] & cur_mask).bit_count()
                        if score > best_score:
                            best_score = score
                            best_v = v
                    mask ^= lb
                
                if best_v != -1:
                    cur_mask = (cur_mask & ~(1 << u)) | (1 << best_v)
                    improved = True
                    not_in_clique = ((1 << m) - 1) & ~cur_mask
                    if cur_mask.bit_count() > best_size:
                        best_size = cur_mask.bit_count()
                        best_mask = cur_mask
        
        # Try 2-swap moves
        if time.time() < deadline - 2:
            improved2 = True
            max_rounds2 = 50
            round2 = 0
            while improved2 and time.time() < deadline - 2 and round2 < max_rounds2:
                improved2 = False
                round2 += 1
                
                clique_list = [i for i in range(m) if (cur_mask >> i) & 1]
                not_in_clique = ((1 << m) - 1) & ~cur_mask
                
                for i in range(len(clique_list)):
                    if time.time() > deadline - 2:
                        break
                    u1 = clique_list[i]
                    for j in range(i + 1, len(clique_list)):
                        if time.time() > deadline - 2:
                            break
                        u2 = clique_list[j]
                        clique_wo_u1u2 = cur_mask & ~(1 << u1) & ~(1 << u2)
                        
                        # Find two vertices v1, v2 not in clique that form a clique with clique_wo_u1u2
                        # and with each other
                        candidates_for_swap = []
                        mask = not_in_clique
                        while mask:
                            lb = mask & (-mask)
                            v = lb.bit_length() - 1
                            if (adj[v] & clique_wo_u1u2) == clique_wo_u1u2:
                                candidates_for_swap.append(v)
                            mask ^= lb
                        
                        # Try pairs from candidates_for_swap
                        for a in range(len(candidates_for_swap)):
                            if time.time() > deadline - 2:
                                break
                            v1 = candidates_for_swap[a]
                            for b in range(a + 1, len(candidates_for_swap)):
                                if time.time() > deadline - 2:
                                    break
                                v2 = candidates_for_swap[b]
                                # v1 and v2 must be adjacent
                                if (adj[v1] >> v2) & 1:
                                    # Valid 2-swap
                                    new_mask = (cur_mask & ~(1 << u1) & ~(1 << u2)) | (1 << v1) | (1 << v2)
                                    if new_mask.bit_count() > cur_mask.bit_count():
                                        cur_mask = new_mask
                                        improved2 = True
                                        not_in_clique = ((1 << m) - 1) & ~cur_mask
                                        if cur_mask.bit_count() > best_size:
                                            best_size = cur_mask.bit_count()
                                            best_mask = cur_mask
                                        break
                            if improved2:
                                break
                        if improved2:
                            break
                    if improved2:
                        break
        
        if cur_mask.bit_count() > best_size:
            best_size = cur_mask.bit_count()
            best_mask = cur_mask
    
    # Convert mask to list
    best_set = [candidates[i] for i in range(m) if (best_mask >> i) & 1]
    
    # Ensure 255 is in the result
    if 255 not in best_set:
        best_set.append(255)
    
    return best_set
