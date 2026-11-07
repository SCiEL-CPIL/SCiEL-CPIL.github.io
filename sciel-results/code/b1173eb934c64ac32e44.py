"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 256

    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    n_nodes = N  # Nodes 0..255 correspond to values 1..256
    weights = [i + 1 for i in range(n_nodes)]
    
    # Build coprime adjacency as bitmasks
    adj_mask = [0] * n_nodes
    for i in range(n_nodes):
        vi = i + 1
        for j in range(i + 1, n_nodes):
            vj = j + 1
            if math.gcd(vi, vj) == 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Precompute weight sum lookup table using 8-bit chunks
    CHUNK_SIZE = 8
    N_CHUNKS = (n_nodes + CHUNK_SIZE - 1) // CHUNK_SIZE  # 32 chunks
    wsum_table = [[0] * 256 for _ in range(N_CHUNKS)]
    for c in range(N_CHUNKS):
        base = c * CHUNK_SIZE
        for byte_val in range(256):
            s = 0
            for b in range(CHUNK_SIZE):
                if byte_val & (1 << b):
                    idx = base + b
                    if idx < n_nodes:
                        s += weights[idx]
            wsum_table[c][byte_val] = s

    def mask_weight_sum(mask):
        """Compute sum of weights for nodes in mask using precomputed table."""
        total = 0
        for c in range(N_CHUNKS):
            byte_val = (mask >> (c * CHUNK_SIZE)) & 0xFF
            total += wsum_table[c][byte_val]
        return total

    # Greedy heuristic to get initial lower bound
    def greedy_clique():
        P = (1 << n_nodes) - 1
        R = 0
        w = 0
        while P:
            best_node = -1
            best_w = -1
            pm = P
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_node = idx
                pm ^= lsb
            
            if best_node == -1:
                break
            
            R |= (1 << best_node)
            w += weights[best_node]
            P = P & adj_mask[best_node] & ~(1 << best_node)
        
        return R, w

    init_R, init_w = greedy_clique()
    global_best_weight = init_w
    global_best_mask = init_R

    # Local search refinement: try to improve by swaps
    def local_search(R_mask):
        """Try 1-for-1, 1-for-2, 2-for-1 swaps to improve the solution."""
        current_w = mask_weight_sum(R_mask)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Build R_set for quick access
            R_set = set()
            mask = R_mask
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                R_set.add(idx)
                mask ^= lsb
            
            # --- 1-for-1 swap: remove v from R, add u not in R ---
            found_improvement = False
            for v in list(R_set):
                R_minus_v_mask = R_mask & ~(1 << v)
                # u must be adjacent to all in R_minus_v_mask
                compatible = (1 << n_nodes) - 1
                for u_idx in R_set:
                    if u_idx != v:
                        compatible &= adj_mask[u_idx]
                compatible &= ~R_mask  # u not already in R
                compatible &= ~(1 << v)  # u != v
                
                # Find best u in compatible
                best_u = -1
                best_gain = -1
                cm = compatible
                while cm:
                    lsb = cm & -cm
                    u = lsb.bit_length() - 1
                    gain = weights[u] - weights[v]
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                    cm ^= lsb
                
                if best_u != -1 and best_gain > 0:
                    R_mask = (R_mask & ~(1 << v)) | (1 << best_u)
                    current_w += best_gain
                    improved = True
                    found_improvement = True
                    break
            
            if found_improvement:
                continue  # Restart search after improvement
            
            # --- 1-for-2 swap: remove v from R, add u1 and u2 not in R ---
            found_improvement = False
            for v in list(R_set):
                R_minus_v_mask = R_mask & ~(1 << v)
                # Candidates must be adjacent to all in R_minus_v_mask
                compatible = (1 << n_nodes) - 1
                for u_idx in R_set:
                    if u_idx != v:
                        compatible &= adj_mask[u_idx]
                compatible &= ~R_mask  # not in R
                compatible &= ~(1 << v)  # != v
                
                # Collect candidate list
                candidates = []
                cm = compatible
                while cm:
                    lsb = cm & -cm
                    u = lsb.bit_length() - 1
                    candidates.append(u)
                    cm ^= lsb
                
                # Try pairs from candidates (sorted by weight desc)
                candidates.sort(key=lambda x: weights[x], reverse=True)
                max_candidates = min(len(candidates), 50)
                best_pair_gain = -1
                best_u1 = -1
                best_u2 = -1
                
                for i in range(max_candidates):
                    u1 = candidates[i]
                    adj_u1 = adj_mask[u1]
                    for j in range(i + 1, max_candidates):
                        u2 = candidates[j]
                        if (adj_u1 >> u2) & 1:
                            gain = weights[u1] + weights[u2] - weights[v]
                            if gain > best_pair_gain:
                                best_pair_gain = gain
                                best_u1 = u1
                                best_u2 = u2
                
                if best_u1 != -1 and best_pair_gain > 0:
                    R_mask = (R_mask & ~(1 << v)) | (1 << best_u1) | (1 << best_u2)
                    current_w += best_pair_gain
                    improved = True
                    found_improvement = True
                    break
            
            if found_improvement:
                continue  # Restart search after improvement
            
            # --- 2-for-1 swap: remove v1, v2 from R, add u not in R ---
            found_improvement = False
            R_list = list(R_set)
            n_r = len(R_list)
            for i in range(n_r):
                v1 = R_list[i]
                for j in range(i + 1, n_r):
                    v2 = R_list[j]
                    # u must be adjacent to all in R minus v1 and v2
                    compatible = (1 << n_nodes) - 1
                    for u_idx in R_set:
                        if u_idx != v1 and u_idx != v2:
                            compatible &= adj_mask[u_idx]
                    compatible &= ~R_mask
                    compatible &= ~(1 << v1) & ~(1 << v2)
                    
                    best_u = -1
                    best_gain = -1
                    cm = compatible
                    while cm:
                        lsb = cm & -cm
                        u = lsb.bit_length() - 1
                        gain = weights[u] - weights[v1] - weights[v2]
                        if gain > best_gain:
                            best_gain = gain
                            best_u = u
                        cm ^= lsb
                    
                    if best_u != -1 and best_gain > 0:
                        R_mask = (R_mask & ~(1 << v1) & ~(1 << v2)) | (1 << best_u)
                        current_w += best_gain
                        improved = True
                        found_improvement = True
                        break
                if found_improvement:
                    break
            
            if found_improvement:
                continue  # Restart search after improvement
            
            # No improvement found
            break
        
        return R_mask, current_w

    global_best_mask, global_best_weight = local_search(global_best_mask)

    # Branch and bound with Bron-Kerbosch style pivot
    sys.setrecursionlimit(10000)
    
    def bk_search(R_mask, P_mask, X_mask, current_weight):
        nonlocal global_best_weight, global_best_mask
        
        if time.time() > deadline:
            return
            
        if not P_mask:
            if current_weight > global_best_weight:
                global_best_weight = current_weight
                global_best_mask = R_mask
            return
            
        # Upper bound: current_weight + sum of all weights in P
        bound = current_weight + mask_weight_sum(P_mask)
        if bound <= global_best_weight:
            return
            
        # Pivot selection: pick node in P with max degree in P
        pivot_idx = -1
        max_deg = -1
        pm = P_mask
        while pm:
            lsb = pm & -pm
            idx = lsb.bit_length() - 1
            deg = (P_mask & adj_mask[idx]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot_idx = idx
            pm ^= lsb
        
        if pivot_idx == -1:
            lsb = P_mask & -P_mask
            pivot_idx = lsb.bit_length() - 1
        
        # Branch on vertices in P that are NOT adjacent to pivot
        verts_to_iter = P_mask & ~adj_mask[pivot_idx]
        
        # Collect and sort by weight descending for better pruning
        verts_list = []
        vm = verts_to_iter
        while vm:
            lsb = vm & -vm
            idx = lsb.bit_length() - 1
            verts_list.append(idx)
            vm ^= lsb
        
        verts_list.sort(key=lambda x: weights[x], reverse=True)
        
        for v in verts_list:
            if time.time() > deadline:
                return
            
            new_R = R_mask | (1 << v)
            new_P = P_mask & adj_mask[v] & ~(1 << v)
            new_X = X_mask & adj_mask[v] & ~(1 << v)
            new_weight = current_weight + weights[v]
            
            bk_search(new_R, new_P, new_X, new_weight)
            
            # Update P and X for next iteration
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
    
    # Initial call
    initial_P = (1 << n_nodes) - 1
    bk_search(0, initial_P, 0, 0)
    
    # Final local search polish on the best found
    global_best_mask, global_best_weight = local_search(global_best_mask)
    
    # Extract result
    result = []
    mask = global_best_mask
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        mask ^= lsb
    
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback: return 1 and all primes (valid but suboptimal)
                return [1] + primes
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
