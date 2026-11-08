"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def strassen_2x2x2_terms():
        terms = []
        u1 = {(0,0): 1, (1,1): 1}; v1 = {(0,0): 1, (1,1): 1}; w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        u2 = {(1,0): 1, (1,1): 1}; v2 = {(0,0): 1}; w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        u3 = {(0,0): 1}; v3 = {(0,1): 1, (1,1): -1}; w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        u4 = {(1,1): 1}; v4 = {(1,0): 1, (0,0): -1}; w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        u5 = {(0,0): 1, (0,1): 1}; v5 = {(1,1): 1}; w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        u6 = {(1,0): 1, (0,0): -1}; v6 = {(0,0): 1, (0,1): 1}; w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        u7 = {(0,1): 1, (1,1): -1}; v7 = {(1,0): 1, (1,1): 1}; w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        return terms

    def build_base_decomposition():
        """Build the rank-84 base decomposition."""
        strassen_terms = strassen_2x2x2_terms()
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1, 2, 3], [4]]
        k_groups = [[0, 1, 2, 3], [4, 5]]
        
        total_rank = 0
        block_decomps = []
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 4 and sk == 4:
                        sub_blocks = []
                        j_sub = [jg[0:2], jg[2:4]]
                        k_sub = [kg[0:2], kg[2:4]]
                        for js in j_sub:
                            for ks in k_sub:
                                r = 7
                                sub_blocks.append((ig, js, ks, r, 2, 2, 2))
                                total_rank += r
                        block_decomps.append(('strassen_grid', sub_blocks))
                    elif si == 2 and sj == 4 and sk == 2:
                        sub_blocks = []
                        j_sub = [jg[0:2], jg[2:4]]
                        for js in j_sub:
                            r = 7
                            sub_blocks.append((ig, js, kg, r, 2, 2, 2))
                            total_rank += r
                        block_decomps.append(('strassen_pair', sub_blocks))
                    else:
                        r = si * sj * sk
                        total_rank += r
                        block_decomps.append(('simple', (ig, jg, kg, r, si, sj, sk)))
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        for block_type, data in block_decomps:
            if block_type in ('strassen_grid', 'strassen_pair'):
                for (ig, jg, kg, r, si, sj, sk) in data:
                    for (u_pat, v_pat, w_pat) in strassen_terms:
                        for (li, lj), val_u in u_pat.items():
                            gi, gj = ig[li], jg[lj]
                            U[gi * m + gj, col] = val_u
                        for (lj, lk), val_v in v_pat.items():
                            gj, gk = jg[lj], kg[lk]
                            V[gj * p + gk, col] = val_v
                        for (lk, li), val_w in w_pat.items():
                            gk, gi = kg[lk], ig[li]
                            W[gk * n + gi, col] = val_w
                        col += 1
            else:
                ig, jg, kg, r, si, sj, sk = data
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj, col] = 1
                            V[gj * p + gk, col] = 1
                            W[gk * n + gi, col] = 1
                            col += 1
        
        return U, V, W, total_rank

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(actual - 1.0) > 1e-7:
                        return False
        return True

    def try_rank_reduction(U, V, W, current_rank):
        """Try to reduce rank by finding linear dependencies."""
        if time.time() > deadline:
            return None
        
        # Try to find a lower-rank decomposition by column elimination
        # For each column, check if it can be expressed as a linear combination of others
        # This is a heuristic approach
        
        best_rank = current_rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try removing columns one at a time and checking if decomposition still holds
        # This is O(rank^2 * n*m*p) which might be too slow for rank 84
        # Instead, try a randomized approach
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Try to find a random linear combination of columns that preserves the tensor
            # We'll try to merge two columns
            if current_rank <= 2:
                break
                
            # Pick two random columns to try to merge
            c1 = random.randint(0, current_rank - 1)
            c2 = random.randint(0, current_rank - 1)
            if c1 == c2:
                continue
            
            # Try to see if column c2 can be expressed as a multiple of column c1
            # in all three matrices simultaneously
            U_c1 = U[:, c1]
            U_c2 = U[:, c2]
            V_c1 = V[:, c1]
            V_c2 = V[:, c2]
            W_c1 = W[:, c1]
            W_c2 = W[:, c2]
            
            # Check if they are proportional
            if np.allclose(U_c1, 0) and np.allclose(U_c2, 0):
                # Both zero in U, can remove one
                pass
            else:
                # Try to find alpha such that U_c2 ≈ alpha * U_c1
                mask = np.abs(U_c1) > 1e-10
                if np.any(mask):
                    alphas = U_c2[mask] / U_c1[mask]
                    if np.allclose(alphas, alphas[0], atol=1e-8):
                        alpha = alphas[0]
                        # Check V and W
                        if np.allclose(V_c2, alpha * V_c1, atol=1e-8) and np.allclose(W_c2, alpha * W_c1, atol=1e-8):
                            # Can remove column c2
                            new_U = np.delete(U, c2, axis=1)
                            new_V = np.delete(V, c2, axis=1)
                            new_W = np.delete(W, c2, axis=1)
                            new_rank = current_rank - 1
                            if verify(new_U, new_V, new_W, new_rank):
                                U, V, W = new_U, new_V, new_W
                                current_rank = new_rank
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                best_rank = current_rank
        
        return U, V, W, current_rank

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Build base decomposition
    U, V, W, rank = build_base_decomposition()
    
    # Verify base
    if verify(U, V, W, rank):
        # Try to reduce rank
        U, V, W, rank = try_rank_reduction(U, V, W, rank)
        
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
