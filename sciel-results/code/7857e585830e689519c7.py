import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def build_tensor():
        """Build the multiplication tensor as a 3D array."""
        T = np.zeros((n_rows, m_rows, p_rows), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T
    
    def verify(U, V, W):
        """Verify decomposition correctness."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Compute the decomposed tensor
        # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for efficiency
        # T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        # But this is O(rank * n_rows * m_rows * p_rows) which might be slow
        # Let's do it in a loop for safety
        T_decomp = np.zeros((n_rows, m_rows, p_rows), dtype=float)
        for r in range(rank):
            T_decomp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        T_true = build_tensor()
        return np.allclose(T_decomp, T_true, atol=1e-9)
    
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # Strassen
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        # Last col of B_T
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}
    
    def try_reduce_rank(U, V, W, max_iters=200):
        """Try to reduce rank by finding linear dependencies and removing terms."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Try to find a term that can be removed
            # A term r is redundant if the tensor it contributes can be expressed
            # as a linear combination of the other terms.
            # 
            # The contribution of term r is: u_r ⊗ v_r ⊗ w_r (outer product)
            # We need to check if this is in the span of {u_s ⊗ v_s ⊗ w_s : s ≠ r}
            #
            # This is equivalent to checking if there exist coefficients c_s such that:
            # u_r ⊗ v_r ⊗ w_r = sum_{s≠r} c_s * (u_s ⊗ v_s ⊗ w_s)
            #
            # This is a linear system in the c_s variables.
            # Flatten both sides and solve.
            
            # For efficiency, try random subsets of terms to remove
            # Start with trying to remove one term at a time
            
            removed = False
            for r in range(rank):
                if time.time() > deadline - 10:
                    break
                
                # Build the linear system
                # LHS: flattened u_r ⊗ v_r ⊗ w_r
                lhs = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).flatten()
                
                # RHS: matrix where column s is flattened u_s ⊗ v_s ⊗ w_s for s ≠ r
                # This is a (n_rows*m_rows*p_rows) x (rank-1) matrix
                # Too large to build explicitly for rank=50 (6*27*18 = 2916 rows, 49 cols)
                # Actually that's manageable!
                
                if rank - 1 <= 0:
                    continue
                
                # Build RHS matrix
                rhs_cols = []
                for s in range(rank):
                    if s == r:
                        continue
                    col = np.outer(np.outer(U[:, s], V[:, s]), W[:, s]).flatten()
                    rhs_cols.append(col)
                
                if not rhs_cols:
                    continue
                
                rhs_matrix = np.column_stack(rhs_cols)  # shape (2916, rank-1)
                
                # Solve rhs_matrix * c = lhs
                try:
                    c, residuals, rank_solve, sv = np.linalg.lstsq(rhs_matrix, lhs, rcond=None)
                except:
                    continue
                
                # Check if the solution is exact
                residual = np.linalg.norm(rhs_matrix @ c - lhs)
                if residual < 1e-8:
                    # Term r can be removed
                    # Remove it from U, V, W
                    mask = [s for s in range(rank) if s != r]
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    
                    # Verify
                    T_decomp = np.zeros((n_rows, m_rows, p_rows), dtype=float)
                    for rr in range(U_new.shape[1]):
                        T_decomp += np.outer(np.outer(U_new[:, rr], V_new[:, rr]), W_new[:, rr])
                    T_true = build_tensor()
                    if np.allclose(T_decomp, T_true, atol=1e-8):
                        U, V, W = U_new, V_new, W_new
                        rank = U.shape[1]
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = rank
                        removed = True
                        break
            
            if not removed:
                break
        
        return best_U, best_V, best_W, best_rank
    
    def try_pairwise_merge(U, V, W, max_attempts=100):
        """Try to merge pairs of terms using Strassen-like identities."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank
        
        # For each pair of terms (r1, r2), check if we can replace them
        # with a single term or fewer terms using a linear transformation.
        #
        # The idea: if u_r1 = a*u_r2 + b*u_other, etc., we might be able to
        # combine them.
        #
        # More practically: check if the sum of two rank-1 tensors can be
        # expressed as a single rank-1 tensor (unlikely) or if there's a
        # 2-to-1 reduction.
        
        # Alternative: try to find 3 terms that can be replaced by 2
        # using a Strassen-like identity.
        
        # For now, just try removing individual terms (already done in try_reduce_rank)
        # and try a random perturbation approach.
        
        return best_U, best_V, best_W, best_rank
    
    # Start with the rank-50 solution
    sol = current_solution()
    U = np.array(sol["U"], dtype=float)
    V = np.array(sol["V"], dtype=float)
    W = np.array(sol["W"], dtype=float)
    
    # Verify initial solution
    if not verify(U.tolist(), V.tolist(), W.tolist()):
        # Should not happen, but return as-is if broken
        return sol
    
    # Try to reduce rank
    U_reduced, V_reduced, W_reduced, new_rank = try_reduce_rank(U, V, W)
    
    if new_rank < U.shape[1]:
        # Convert back to list format
        U_out = U_reduced.tolist()
        V_out = V_reduced.tolist()
        W_out = W_reduced.tolist()
        
        # Final verification
        if verify(U_out, V_out, W_out):
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Return original if no improvement
    return sol

