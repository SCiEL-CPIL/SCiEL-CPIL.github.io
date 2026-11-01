"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools
import sys

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Try to find a rank-minimal decomposition for 2x2x9.
    # We know the rank is at most 36 (schoolbook).
    # We will use a randomized greedy algorithm (similar to the "rank minimization" heuristic
    # often used in tensor decomposition searches) to try to reduce the rank below 36.
    # Since 2x2x9 is small, we can afford a relatively intensive search within the time limit.
    
    n, m, p = 2, 2, 9
    target_dim_u = n * m  # 4
    target_dim_v = m * p  # 18
    target_dim_w = p * n  # 18
    
    # The tensor T is of shape (4, 18, 18).
    # T[i, j, k] = 1 if (i, j, k) corresponds to a valid multiplication term.
    # i = a * m + b  (a in 0..1, b in 0..1)
    # j = b * p + c  (b in 0..1, c in 0..8)
    # k = c * n + a  (c in 0..8, a in 0..1)
    # Term is non-zero if the indices are consistent:
    # a from i, b from i; b from j, c from j; c from k, a from k.
    # So for a given (i, j, k), we check:
    # a1 = i // m, b1 = i % m
    # b2 = j // p, c1 = j % p
    # c2 = k // n, a2 = k % n
    # It's valid if a1 == a2 and b1 == b2.
    
    T = np.zeros((target_dim_u, target_dim_v, target_dim_w))
    for i in range(target_dim_u):
        a1 = i // m
        b1 = i % m
        for j in range(target_dim_v):
            b2 = j // p
            c1 = j % p
            if b1 != b2: continue
            for k in range(target_dim_w):
                c2 = k // n
                a2 = k % n
                if c1 != c2 or a1 != a2: continue
                T[i, j, k] = 1.0

    # We want to find rank r such that T = sum_r U_r V_r W_r
    # U_r is a vector of dim 4, V_r is a vector of dim 18, W_r is a vector of dim 18.
    # This is equivalent to finding a set of r "atoms" (rank-1 tensors) that sum to T.
    
    # A standard heuristic is to pick a random vector, project T onto it to get the next factor,
    # subtract, and repeat. We can use a "greedy" approach with random restarts.
    
    best_rank = n * m * p  # 36
    best_factors = None
    start_time = time.time()
    time_limit = 100  # seconds, leave 20s buffer for evaluator overhead
    
    np.random.seed(42)
    
    # Pre-compute slices for speed
    # T_flat is not quite right, we need to handle the structure.
    
    def try_decomposition(max_rank, iterations=50):
        """Try to find a decomposition of rank <= max_rank."""
        for _ in range(iterations):
            U_list = []
            V_list = []
            W_list = []
            T_curr = T.copy()
            
            for r in range(max_rank):
                # Random direction in U-space
                u_rand = np.random.randn(target_dim_u)
                
                # Project T_curr onto u_rand to get V_curr
                # V_curr[j] = sum_i T_curr[i, j, :] dot u_rand[i] ? No.
                # T = sum U_r V_r W_r
                # T[i, j, k] = U_r[i] V_r[j] W_r[k]
                # For fixed U, T is a matrix M[j, k] = sum_i T[i, j, k] U[i].
                # This M should be rank 1: M = V W^T.
                
                M = np.tensordot(u_rand, T_curr, axes=([0], [0]))
                # M is (18, 18)
                
                if np.linalg.norm(M) < 1e-9:
                    # T_curr is orthogonal to u_rand? Unlikely if T_curr != 0.
                    # If T_curr is close to 0, we are done.
                    break
                
                # Find the best rank-1 approximation of M
                # This is just the largest singular value decomposition.
                try:
                    U_m, s, Vt = np.linalg.svd(M, full_matrices=False)
                    if len(s) == 0 or s[0] < 1e-9:
                        break
                    # V is the first right singular vector, W is the first left singular vector (transposed)
                    # M approx s[0] * V[:, 0] * U_m[0, :]
                    v_vec = Vt[0, :]  # shape (18,)
                    w_vec = U_m[:, 0] # shape (18,)
                    scale = s[0]
                except:
                    return None
                
                # Update T_curr
                # T_curr -= U_r (outer) V_r (outer) W_r
                # U_r is u_rand. But we can scale u_rand to make the subtraction exact for the first step?
                # Actually, we can absorb the scale into the vectors.
                # Let U_r = u_rand / ||u_rand|| ? No, let's just use u_rand.
                # The term is u_rand[None, :, None] * v_vec[None, None, :] * w_vec[:, None, None]
                # Wait, the projection was:
                # M[j, k] = sum_i T_curr[i, j, k] u_rand[i]
                # If T_curr = U V W^T + T_rest, and we pick u_rand parallel to U, then M = ||U||^2 V W^T.
                # So V and W are the singular vectors.
                # We can set U_r = u_rand.
                # Then V_r should be v_vec * (some factor) and W_r should be w_vec * (some factor).
                # Specifically, we want U_r V_r W_r to match the projection.
                # Actually, it's safer to normalize U_r to 1, and adjust V and W.
                
                norm_u = np.linalg.norm(u_rand)
                if norm_u < 1e-9: continue
                u_vec = u_rand / norm_u
                
                # Rescale V and W to match the magnitude
                # The term added to T is u_vec (outer) v_vec_scaled (outer) w_vec_scaled
                # We want the projection of this term onto u_rand to be M.
                # <u_rand, u_vec> v_vec_scaled w_vec_scaled^T = M
                # (1/norm_u) v_vec_scaled w_vec_scaled^T = M  (since <u_rand, u_vec> = 1)
                # v_vec_scaled w_vec_scaled^T = norm_u * M
                # Since M = s0 * v_vec * w_vec^T, we have v_vec_scaled = v_vec * sqrt(s0 * norm_u)
                # and w_vec_scaled = w_vec * sqrt(s0 / norm_u) ? No.
                # Let v_scaled = alpha * v_vec, w_scaled = beta * w_vec.
                # alpha beta v_vec w_vec^T = norm_u * s0 v_vec w_vec^T
                # alpha beta = norm_u * s0.
                # We can just pick alpha = 1, beta = norm_u * s0.
                # Or better, keep them balanced.
                # Actually, it's simpler to just use the SVD factors directly and normalize U to 1.
                # Let's set:
                # U_r = u_vec
                # V_r = v_vec
                # W_r = w_vec * (s0 * norm_u)
                
                # Check subtraction
                term = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec * (s0 * norm_u))
                T_curr -= term
                
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec * (s0 * norm_u))
                
                if np.linalg.norm(T_curr) < 1e-9:
                    # Success
                    U_arr = np.array(U_list).T  # (4, r)
                    V_arr = np.array(V_list).T  # (18, r)
                    W_arr = np.array(W_list).T  # (18, r)
                    return U_arr, V_arr, W_arr
                    
            return None
        
    # Try ranks from low to high
    # Lower bound: max(target_dim_u, target_dim_v, target_dim_w) = 18.
    # But actually rank is at least max(4, 18, 18) = 18.
    # Also rank is at least the dimension of the support.
    # Let's try 18, 19, ..., 36.
    
    for r in range(18, 37):
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
            
        result = try_decomposition(r, iterations=20)
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Verify
            T_check = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
            if np.allclose(T_check, T, atol=1e-6):
                best_rank = r
                best_factors = (U_arr, V_arr, W_arr)
                break
                
    if best_factors is None:
        # Fallback to schoolbook
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        U_arr, V_arr, W_arr = best_factors
        # Convert to lists of lists as required
        U_list = U_arr.tolist()
        V_list = V_arr.tolist()
        W_list = W_arr.tolist()
        
        # Post-process to remove small values (due to floating point noise)
        # and convert to integers if close
        def clean_matrix(mat):
            out = []
            for row in mat:
                new_row = []
                for val in row:
                    if abs(val) < 1e-9:
                        new_row.append(0)
                    else:
                        # Check if close to integer or half-integer
                        nearest = round(val * 2) / 2
                        if abs(val - nearest) < 1e-9:
                            new_row.append(nearest)
                        else:
                            new_row.append(val)
                out.append(new_row)
            return out

        return {"U": clean_matrix(U_list), "V": clean_matrix(V_list), "W": clean_matrix(W_list)}

# EVOLVE-BLOCK-END
