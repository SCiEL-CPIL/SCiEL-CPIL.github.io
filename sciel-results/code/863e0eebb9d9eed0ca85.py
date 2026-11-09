import time
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p
    
    def verify_decomp(U, V, W, rank):
        """
        Verifies if the given U, V, W matrices exactly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Condition: For all i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def round_to_half(arr):
        return np.round(arr * 2) / 2.0

    def get_rank_17_solution():
        """
        Returns a known valid rank-17 decomposition.
        """
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Term 1
        U[0, 0] += 1; U[4, 0] += 1
        V[0, 0] += 1; V[4, 0] += 1
        W[0, 0] += 1; W[3, 0] += 1
        
        # Term 2
        U[3, 1] += 1; U[4, 1] += 1
        V[0, 1] += 1
        W[1, 1] += 1; W[3, 1] -= 1
        
        # Term 3
        U[0, 2] += 1
        V[1, 2] += 1; V[4, 2] -= 1
        W[2, 2] += 1; W[3, 2] += 1
        
        # Term 4
        U[4, 3] += 1
        V[3, 3] += 1; V[0, 3] -= 1
        W[0, 3] += 1; W[1, 3] += 1
        
        # Term 5
        U[0, 4] += 1; U[1, 4] += 1
        V[4, 4] += 1
        W[0, 4] -= 1; W[2, 4] += 1
        
        # Term 6
        U[3, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        W[3, 5] += 1
        
        # Term 7
        U[1, 6] += 1; U[4, 6] -= 1
        V[3, 6] += 1; V[4, 6] += 1
        W[0, 6] += 1
        
        # Term 8: A00B02
        U[0, 7] += 1
        V[2, 7] += 1
        W[4, 7] += 1
        
        # Term 9: A01B12
        U[1, 8] += 1
        V[5, 8] += 1
        W[4, 8] += 1
        
        # Term 10: A10B02
        U[3, 9] += 1
        V[2, 9] += 1
        W[5, 9] += 1
        
        # Term 11: A11B12
        U[4, 10] += 1
        V[5, 10] += 1
        W[5, 10] += 1
        
        # Term 12: A02B20
        U[2, 11] += 1
        V[6, 11] += 1
        W[0, 11] += 1
        
        # Term 13: A02B21
        U[2, 12] += 1
        V[7, 12] += 1
        W[2, 12] += 1
        
        # Term 14: A02B22
        U[2, 13] += 1
        V[8, 13] += 1
        W[4, 13] += 1
        
        # Term 15: A12B20
        U[5, 14] += 1
        V[6, 14] += 1
        W[1, 14] += 1
        
        # Term 16: A12B21
        U[5, 15] += 1
        V[7, 15] += 1
        W[3, 15] += 1
        
        # Term 17: A12B22
        U[5, 16] += 1
        V[8, 16] += 1
        W[5, 16] += 1
        
        return U, V, W

    def try_merge_terms(U17, V17, W17):
        """
        Try to merge pairs of terms from the rank-17 solution to get rank-16.
        """
        U, V, W = U17.copy(), V17.copy(), W17.copy()
        rank17 = 17
        
        # Try merging each pair of terms
        for i in range(rank17):
            for j in range(i + 1, rank17):
                if time.time() > deadline - 10:
                    return None
                
                # Create a candidate by merging terms i and j
                U_cand = U.copy()
                V_cand = V.copy()
                W_cand = W.copy()
                
                # Zero out term j
                U_cand[:, j] = 0
                V_cand[:, j] = 0
                W_cand[:, j] = 0
                
                # Try to find coefficients a, b such that:
                # a * term_i + b * term_j can replace both terms
                # This is a nonlinear problem, so we'll use optimization
                
                # Initialize with a=1, b=0 (just term i)
                def merge_error(params):
                    a, b = params
                    # The merged term is a*term_i + b*term_j
                    # We need: sum_r (a*U_i[r] + b*U_j[r]) * (a*V_i[r] + b*V_j[r]) * (a*W_i[r] + b*W_j[r]) = 1
                    # But we also need to ensure other terms are not affected
                    # Actually, we're replacing two terms with one, so we need:
                    # (a*U_i + b*U_j) * (a*V_i + b*V_j) * (a*W_i + b*W_j) should contribute 1
                    # while the original two terms contributed 1+1=2? No, that's not right.
                    
                    # Let me reconsider. We have 17 terms, each contributing to the tensor.
                    # If we merge terms i and j into a single term, we need the new term
                    # to have the same "effect" as the sum of the two original terms.
                    # This is not straightforward.
                    
                    # Alternative approach: just try random perturbations and see if we can
                    # reduce the rank by making one term redundant.
                    return 0.0
                
                # This approach is getting complicated. Let's try a different strategy:
                # Use the rank-17 solution as a starting point and try to optimize
                # a rank-16 solution that's close to it.
                
                break
            if time.time() > deadline - 10:
                return None
        
        return None

    def try_rank_16_direct():
        """
        Attempts to find a rank-16 decomposition using a smarter approach.
        """
        if time.time() > deadline - 15:
            return None
        
        rank = 16
        U17, V17, W17 = get_rank_17_solution()
        
        # Strategy: Start from rank-17 and try to eliminate one term by adjusting others
        # We'll try each term as a candidate for elimination
        
        for elim_idx in range(17):
            if time.time() > deadline - 5:
                return None
            
            # Create initial guess: use the rank-17 solution without term elim_idx
            U_init = np.zeros((n * m, rank))
            V_init = np.zeros((m * p, rank))
            W_init = np.zeros((p * n, rank))
            
            # Copy terms, skipping elim_idx
            count = 0
            for idx in range(17):
                if idx != elim_idx:
                    U_init[:, count] = U17[:, idx]
                    V_init[:, count] = V17[:, idx]
                    W_init[:, count] = W17[:, idx]
                    count += 1
            
            # Now we have 16 terms, but they don't sum to the correct tensor
            # We need to adjust them to make up for the missing term
            
            def error_func(params):
                U_flat = params[0:n*m*rank]
                V_flat = params[n*m*rank:n*m*rank+m*p*rank]
                W_flat = params[n*m*rank+m*p*rank:]
                
                U_ = U_flat.reshape(n*m, rank)
                V_ = V_flat.reshape(m*p, rank)
                W_ = W_flat.reshape(p*n, rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            s = np.dot(U_[u_idx, :], V_[v_idx, :] * W_[w_idx, :])
                            err += (s - 1.0)**2
                return err
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(error_func, params0, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if res.fun < 1e-10:
                    U_sol = res.x[0:n*m*rank].reshape(n*m, rank)
                    V_sol = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_sol = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    # Try to round to half-integers
                    U_rounded = round_to_half(U_sol)
                    V_rounded = round_to_half(V_sol)
                    W_rounded = round_to_half(W_sol)
                    
                    if verify_decomp(U_rounded, V_rounded, W_rounded, rank):
                        return [U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()]
                        
            except Exception:
                continue
        
        return None

    # Try to find a rank-16 solution
    result_16 = try_rank_16_direct()
    
    if result_16:
        U, V, W = result_16
        return {"U": U, "V": V, "W": W}
        
    # Fall back to the known rank 17 solution
    U17, V17, W17 = get_rank_17_solution()
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
