"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 21 decomposition using Strassen's algorithm for 3 blocks
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_coeffs = [
            {0: 1, 3: 1},       # M1: A00 + A11
            {2: 1, 3: 1},       # M2: A10 + A11
            {0: 1},             # M3: A00
            {3: 1},             # M4: A11
            {0: 1, 1: 1},       # M5: A00 + A01
            {2: 1, 0: -1},      # M6: A10 - A00
            {1: 1, 3: -1}       # M7: A01 - A11
        ]
        
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
            [(0, 0, 1)],                 # M2: B00
            [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
            [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
            [(1, 1, 1)],                 # M5: B11
            [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
            [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
        ]
        
        W_map = [
            [(0, 0, 1), (1, 1, 1)],   # k=0 (M1) -> C00, C11
            [(1, 0, 1), (1, 1, -1)],  # k=1 (M2) -> C10, C11
            [(0, 1, 1), (1, 1, 1)],   # k=2 (M3) -> C01, C11
            [(0, 0, 1), (1, 0, 1)],   # k=3 (M4) -> C00, C10
            [(0, 0, -1), (0, 1, 1)],  # k=4 (M5) -> C00, C01
            [(1, 1, 1)],              # k=5 (M6) -> C11
            [(0, 0, 1)]               # k=6 (M7) -> C00
        ]
        
        for r in range(3):
            for k in range(7):
                col = r * 7 + k
                
                for idx, val in U_coeffs[k].items():
                    U[idx][col] = val
                for j_local, k_local, val in V_coeffs[k]:
                    global_k = 2 * r + k_local
                    v_idx = j_local * p + global_k
                    V[v_idx][col] = val
                for i, j_local, val in W_map[k]:
                    global_k = 2 * r + j_local
                    w_idx = global_k * n + i
                    W[w_idx][col] = val
        return {"U": U, "V": V, "W": W}

    # Attempt to find Rank 20
    rank = 20
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Target tensor T flattened
    # T_{i, j, k} = 1 if i,j,k are valid indices for the multiplication?
    # Actually, the tensor is defined by the equation:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = delta_{?}
    # Let's construct the target vector for the bilinear system.
    # The equation is:
    # For all i in 0..n-1, k in 0..p-1:
    #   C[i, k] = sum_j A[i, j] B[j, k]
    #   sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = A[i, j] B[j, k]
    # This must hold for all A, B.
    # This implies that the tensor T_{(i,j), (j,k), (k,i)} has 1s where the indices match the multiplication structure.
    # Specifically, the coefficient of A[i,j] B[j,k] in the expression for C[i,k] is 1.
    # So, for each r, the term is U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # We need sum_r U[...] V[...] W[...] = 1 if the indices correspond to a valid multiplication term, and 0 otherwise?
    # No, the "otherwise" case doesn't exist. The sum must reconstruct the full tensor.
    # The tensor T has non-zero entries only at specific indices.
    # Let's define the target matrix M of size (U_dim * V_dim) x W_dim? No.
    # Let's use the bilinear form:
    # Let x be the vector of U entries (size U_dim * rank).
    # Let y be the vector of V entries (size V_dim * rank).
    # Let z be the vector of W entries (size W_dim * rank).
    # The constraint is: for all i, j, k:
    #   sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # Wait, is it 1 for all i,j,k?
    # No. C[i,k] = sum_j A[i,j] B[j,k].
    # The term A[i,j] B[j,k] appears in C[i,k].
    # So for a fixed i, j, k, the coefficient in the product expansion is 1.
    # But the decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_k' V[j'*p+k, r] B[j',k]) W[k*n+i, r]?
    # No, the standard decomposition is:
    # C[i,k] = sum_r U_i^r V_k^r W_{i,k}^r?
    # Let's look at the seed's W index: W[k*n+i, r].
    # And U index: i*m+j.
    # And V index: j*p+k.
    # The product term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This term contributes to C[i,k] only if the indices match.
    # In the expansion of C[i,k] = sum_j A[i,j] B[j,k], the term A[i,j] B[j,k] has coefficient 1.
    # So, for each (i, j, k), the sum over r of U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] must equal 1.
    # Are there other constraints?
    # If we pick a different j' or k', does it contribute?
    # The structure of the tensor is such that T_{(i,j), (j,k), (k,i)} = 1.
    # And T_{(i,j), (j',k), (k,i)} = 0 if j != j'?
    # No, the tensor is sparse.
    # The condition is:
    # sum_r U[u, r] V[v, r] W[w, r] = T_{u,v,w}
    # where T_{u,v,w} = 1 if u=i*m+j, v=j'*p+k, w=k'*n+i' and i=i', j=j', k=k'?
    # Actually, for matrix multiplication, the tensor is:
    # T_{(i,j), (j,k), (k,i)} = 1.
    # All other entries are 0.
    # So we have n*m*p = 2*2*6 = 24 equations.
    
    target = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    # We want to find U, V, W of rank R such that:
    # sum_r U[:, :, r] V[:, :, r] W[:, :, r] = target
    # This is a 3-way tensor decomposition.
    
    # Strategy: Random initialization and alternating minimization.
    # Fix W, solve for U and V?
    # The equation is bilinear in (U, V) for fixed W, and bilinear in (U, W) for fixed V.
    # Let's fix V and W, and solve for U.
    # For each r, U[:, r] is a vector.
    # The equation is: sum_r U[:, r] (V[:, r] * W[:, r]) = target_flattened?
    # No, it's a tensor.
    # Let's reshape.
    # For each r, let v_r be V[:, r] and w_r be W[:, r].
    # The term U[:, r] contributes to the tensor via U[:, r] (outer) v_r (outer) w_r.
    # We want sum_r U[:, r] (outer) v_r (outer) w_r = target.
    # This is equivalent to:
    # For each (u, v, w): sum_r U[u, r] V[v, r] W[w, r] = target[u, v, w].
    
    # We can fix V and W, and solve for U.
    # For each u, the equation is:
    # sum_r U[u, r] (sum_v V[v, r] W[w, r] ?) No.
    # For a fixed u, we have equations for all (v, w).
    # sum_r U[u, r] V[v, r] W[w, r] = target[u, v, w].
    # This is a linear system for U[u, :] if V and W are fixed.
    # Let M_u be a matrix of size (V_dim * W_dim) x rank.
    # M_u[v*W_dim + w, r] = V[v, r] * W[w, r].
    # Then M_u @ U[u, :] = target[u, :, :].flatten().
    
    # Similarly for V and W.
    
    best_score = float('inf')
    best_solution = None
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        U_rand = np.random.randn(U_dim, rank)
        V_rand = np.random.randn(V_dim, rank)
        W_rand = np.random.randn(W_dim, rank)
        
        # Normalize to avoid scaling issues
        U_rand /= np.linalg.norm(U_rand)
        V_rand /= np.linalg.norm(V_rand)
        W_rand /= np.linalg.norm(W_rand)
        
        U_curr = U_rand.copy()
        V_curr = V_rand.copy()
        W_curr = W_rand.copy()
        
        # Alternating Minimization
        for it in range(20):
            if time.time() > deadline:
                break
                
            # Solve for U
            for u in range(U_dim):
                # Construct M_u
                # M_u[v*W_dim + w, r] = V_curr[v, r] * W_curr[w, r]
                # Vectorize V and W
                # V_curr is (V_dim, rank)
                # W_curr is (W_dim, rank)
                # We need the outer product of V_curr[v, :] and W_curr[w, :] for each r?
                # No, for each r, the term is V_curr[v, r] * W_curr[w, r].
                # So M_u is (V_dim * W_dim) x rank.
                M_u = np.zeros((V_dim * W_dim, rank))
                for v in range(V_dim):
                    for w in range(W_dim):
                        M_u[v * W_dim + w, :] = V_curr[v, :] * W_curr[w, :]
                
                b_u = target[u, :, :].flatten()
                
                try:
                    U_curr[u, :], residuals, rank_u, sv = np.linalg.lstsq(M_u, b_u, rcond=None)
                except:
                    continue
                    
            # Solve for V
            for v in range(V_dim):
                M_v = np.zeros((U_dim * W_dim, rank))
                for u in range(U_dim):
                    for w in range(W_dim):
                        M_v[u * W_dim + w, :] = U_curr[u, :] * W_curr[w, :]
                b_v = target[:, v, :].flatten()
                try:
                    V_curr[v, :], residuals, rank_v, sv = np.linalg.lstsq(M_v, b_v, rcond=None)
                except:
                    continue
                    
            # Solve for W
            for w in range(W_dim):
                M_w = np.zeros((U_dim * V_dim, rank))
                for u in range(U_dim):
                    for v in range(V_dim):
                        M_w[u * V_dim + v, :] = U_curr[u, :] * V_curr[v, :]
                b_w = target[:, :, w].flatten()
                try:
                    W_curr[w, :], residuals, rank_w, sv = np.linalg.lstsq(M_w, b_w, rcond=None)
                except:
                    continue
            
            # Calculate residual
            # Reconstruct tensor
            T_rec = np.zeros((U_dim, V_dim, W_dim))
            for r in range(rank):
                T_rec += np.einsum('u,v,w->uvw', U_curr[:, r], V_curr[:, r], W_curr[:, r])
            residual = np.linalg.norm(T_rec - target)
            
            if residual < best_score:
                best_score = residual
                best_solution = (U_curr.copy(), V_curr.copy(), W_curr.copy())
            
            if residual < 1e-8:
                break
                
        if best_score < 1e-8:
            break

    if best_solution is not None and best_score < 1e-5:
        U_sol, V_sol, W_sol = best_solution
        # Snap to integers/half-integers
        U_snap = np.round(U_sol * 2) / 2
        V_snap = np.round(V_sol * 2) / 2
        W_snap = np.round(W_sol * 2) / 2
        
        # Verify
        T_rec = np.zeros((U_dim, V_dim, W_dim))
        for r in range(rank):
            T_rec += np.einsum('u,v,w->uvw', U_snap[:, r], V_snap[:, r], W_snap[:, r])
        if np.linalg.norm(T_rec - target) < 1e-5:
            return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }

    # Fallback to Rank 21
    return get_rank_21()

# EVOLVE-BLOCK-END
