"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor M of shape (n*m, m*p, p*n)
    # M[a, b, c] = 1 if A[a] * B[b] contributes to C[c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    M = np.zeros((dim_A, dim_B, dim_C), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1
    
    schoolbook_rank = n * m * p
    
    # Greedy Tensor Decomposition Search
    # We try to decompose M into sum of rank-1 tensors u v w
    # We prefer integer/half-integer solutions.
    
    def try_greedy_decomposition(max_rank):
        residual = M.copy()
        U_list = []
        V_list = []
        W_list = []
        
        # Candidate vectors to try for each step.
        # Since M has entries 0 and 1, and we want small rank, 
        # the coefficients might be integers.
        # Let's try to find u, v, w such that u @ v @ w approximates residual.
        
        for step in range(max_rank):
            if not np.any(residual):
                break
                
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            # Search space for u, v, w
            # To keep it fast, we limit the search to vectors with small integer entries
            # or use a heuristic based on SVD of the residual.
            
            # 1. SVD Heuristic to get directions
            # Reshape residual to 2D matrices for each mode
            R1 = np.tensordot(residual, np.eye(dim_B * dim_C).reshape(dim_B, dim_C, dim_B * dim_C), axes=([1, 2], [0, 1]))
            # This is getting complex. Let's use a simpler heuristic:
            # Sample random integer vectors, or use the dominant left/right singular vectors.
            
            # Let's use a simpler approach: 
            # If the residual is sparse, we might find exact integer matches.
            # But for general matmul, the structure is dense.
            
            # Let's try a limited combinatorial search for the first few terms
            # and then fall back to SVD-guided snapping.
            
            # For speed, we will only attempt this if we haven't exceeded a time limit.
            # Given the 120s limit, we can afford a moderate search.
            
            # Strategy: 
            # 1. Compute SVD of residual unfolded along mode 1, 2, 3.
            # 2. Take the principal vectors.
            # 3. Snap to rationals with small denominators (1, 2, 3, 4).
            # 4. Check if the sum of these snapped vectors is a valid rank-1 term (i.e., does it reduce the residual norm significantly?).
            # 5. Subtract and repeat.
            
            # Unfold 1: (dim_A, dim_B*dim_C)
            U_unf = residual.reshape(dim_A, -1)
            _, _, Vt = np.linalg.svd(U_unf, full_matrices=False)
            u_init = Vt[0] if Vt.shape[0] > 0 else None
            
            if u_init is None:
                break
            
            # Normalize and scale to small integers?
            # Let's try scaling factors
            found_term = False
            for scale in [1, 0.5, 2, 0.25]:
                u_cand = np.round(u_init * scale)
                if np.allclose(u_cand, 0):
                    continue
                
                # Given u_cand, we need to find v, w such that u_cand (v (w)^T) approximates residual
                # This is equivalent to finding best rank-1 approximation of residual given u.
                # Let R_u be residual with u projected out?
                
                # Simpler: Just try random small integer vectors for u, v, w?
                # No, that's too slow.
                
                # Let's stick to the SVD direction and snap to integers/halves.
                # We will test a few candidates for (u, v, w)
                
                # Get v_init and w_init from other modes
                V_unf = residual.reshape(dim_B, -1)
                _, _, Vt2 = np.linalg.svd(V_unf, full_matrices=False)
                v_init = Vt2[0] if Vt2.shape[0] > 0 else None
                
                W_unf = residual.reshape(dim_C, -1)
                _, _, Wt = np.linalg.svd(W_unf, full_matrices=False)
                w_init = Wt[0] if Wt.shape[0] > 0 else None
                
                if v_init is None or w_init is None:
                    continue
                
                # Try snapping u, v, w to small rationals
                candidates = []
                for su in [1, 0.5, 2]:
                    for sv in [1, 0.5, 2]:
                        for sw in [1, 0.5, 2]:
                            u_t = np.round(u_init * su)
                            v_t = np.round(v_init * sv)
                            w_t = np.round(w_init * sw)
                            
                            # Ensure not all zero
                            if np.any(u_t) and np.any(v_t) and np.any(w_t):
                                # Calculate the rank-1 tensor
                                term = np.einsum('a,b,c->abc', u_t, v_t, w_t)
                                # Calculate how much of the residual this term explains
                                # We want to minimize ||residual - term||^2
                                # This is equivalent to maximizing <residual, term> - 0.5 ||term||^2
                                score = np.sum(residual * term) - 0.5 * np.sum(term**2)
                                candidates.append((score, u_t, v_t, w_t, term))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_score, best_u, best_v, best_w, best_term = candidates[0]
                    
                    if best_score > 1e-6: # Significant improvement
                        residual = residual - best_term
                        U_list.append(best_u)
                        V_list.append(best_v)
                        W_list.append(best_w)
                        found_term = True
                        break
                else:
                    break
            
            if not found_term:
                # If we can't find a good integer/half-integer term, break
                # This might mean we are stuck or need non-integer coeffs
                # For safety, we stop greedy search
                break
        
        if not np.allclose(residual, 0):
            return None
        
        return U_list, V_list, W_list

    # Try to find a decomposition better than schoolbook
    # We try ranks from 1 up to schoolbook_rank - 1
    # But doing this iteratively is slow. 
    # Instead, we run the greedy search with a cap on rank.
    
    # Let's try to find *any* decomposition with rank < schoolbook_rank.
    # We will run the greedy search with max_rank = schoolbook_rank - 1.
    # If it fails, we return schoolbook.
    
    # Note: The greedy search above might be too slow if it runs for 62 steps with SVD each time.
    # Optimize: Only run if we hope to find something better.
    # For 3x3x7, schoolbook is 63.
    # If we find one with < 63, it's a win.
    
    # To ensure speed, we limit the number of greedy steps to, say, 50.
    # If it doesn't converge to 0, we abort.
    
    result = try_greedy_decomposition(schoolbook_rank - 1)
    
    if result:
        U_list, V_list, W_list = result
        rank = len(U_list)
        
        # Convert to list of lists format
        U = np.zeros((dim_A, rank))
        V = np.zeros((dim_B, rank))
        W = np.zeros((dim_C, rank))
        
        for r in range(rank):
            U[:, r] = U_list[r]
            V[:, r] = V_list[r]
            W[:, r] = W_list[r]
            
        # Verify decomposition
        # Check if sum U_r V_r W_r == M
        check = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(check, M):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to Schoolbook
    U = [[0] * schoolbook_rank for _ in range(dim_A)]
    V = [[0] * schoolbook_rank for _ in range(dim_B)]
    W = [[0] * schoolbook_rank for _ in range(dim_C)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
