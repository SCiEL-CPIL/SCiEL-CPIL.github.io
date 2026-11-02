"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    def _verify_fast(U, V, W):
        """Faster verification using einsum"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # We need: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _snap_to_half_int(arr):
        """Snap values to nearest half-integer"""
        return np.round(2 * arr) / 2.0

    def _solve_linear_system(A, b):
        """Solve Ax = b using least squares, return None if no exact solution"""
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if it's an exact solution
            if np.allclose(A @ x, b, atol=1e-10):
                return x
        except:
            pass
        return None

    def _try_fixed_W_decomposition(rank, W_pattern):
        """
        Try to find U, V such that for all (i,j,k):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        Fix W to W_pattern, then this becomes a bilinear system in U and V.
        We can vectorize: for each (i,j), we have a linear equation in V given U,
        or vice versa.
        
        Strategy: Fix U to have a simple structure, then solve for V.
        Or use alternating least squares.
        """
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = W_pattern.copy()
        
        # For each (i,j,k), the equation is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is bilinear in U and V. Let's try alternating minimization.
        
        # Initialize U randomly
        np.random.seed(42)
        U = np.random.choice([-1, 0, 1], size=(n*m, rank))
        
        # Iterate: fix U, solve for V; fix V, solve for U
        for iteration in range(50):
            if time.time() - start_time > time_limit:
                return None
            
            # Fix U, solve for V
            # For each (j,p_k) where p_k = k, we need:
            # sum_r V[j*p+k, r] * (sum_i U[i*m+j, r] * W[k*n+i, r]) = 1 for all i,k
            # Wait, that's not quite right. Let me reformulate.
            
            # For fixed U and W, the equation for V[j*p+k, r] is:
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is linear in V for fixed U and W.
            
            # Let's set up the linear system for V.
            # Unknowns: V[j*p+k, r] for j in range(m), k in range(p), r in range(rank)
            # Total unknowns: m * p * rank
            
            # For each (i,j,k), we have one equation:
            # sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1
            
            # This is a system of n*m*p equations in m*p*rank unknowns.
            
            # Build the linear system
            num_equations = n * m * p
            num_unknowns = m * p * rank
            
            A = np.zeros((num_equations, num_unknowns))
            b = np.ones(num_equations)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            # Unknown index for V[j*p+k, r]
                            unk_idx = (j * p + k) * rank + r
                            A[eq_idx, unk_idx] = U[i*m+j, r] * W[k*n+i, r]
                        eq_idx += 1
            
            # Solve for V
            V_flat = _solve_linear_system(A, b)
            if V_flat is None:
                return None
            
            V = V_flat.reshape(m * p, rank)
            
            # Now fix V and W, solve for U
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Linear in U.
            
            A = np.zeros((num_equations, n * m * rank))
            b = np.ones(num_equations)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            # Unknown index for U[i*m+j, r]
                            unk_idx = (i * m + j) * rank + r
                            A[eq_idx, unk_idx] = V[j*p+k, r] * W[k*n+i, r]
                        eq_idx += 1
            
            U_flat = _solve_linear_system(A, b)
            if U_flat is None:
                return None
            
            U = U_flat.reshape(n * m, rank)
        
        # Verify the solution
        if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
            # Snap to half-integers
            U = _snap_to_half_int(U)
            V = _snap_to_half_int(V)
            W = _snap_to_half_int(W)
            
            # Re-verify after snapping
            if _verify_fast(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    def _try_systematic_search(rank):
        """
        Systematic search for rank-r decomposition.
        Try different W patterns and solve for U, V.
        """
        # Try different W patterns
        # Pattern 1: W[k*n+i, r] = 1 for all k,i,r
        W_pattern1 = np.ones((p*n, rank))
        result = _try_fixed_W_decomposition(rank, W_pattern1)
        if result is not None:
            return result
        
        # Pattern 2: W[k*n+i, r] = (-1)^k
        W_pattern2 = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W_pattern2[k*n+i, r] = (-1)**k
        result = _try_fixed_W_decomposition(rank, W_pattern2)
        if result is not None:
            return result
        
        # Pattern 3: W[k*n+i, r] = 1 if k==0 else 0
        W_pattern3 = np.zeros((p*n, rank))
        for i in range(n):
            for r in range(rank):
                W_pattern3[0*n+i, r] = 1
        result = _try_fixed_W_decomposition(rank, W_pattern3)
        if result is not None:
            return result
        
        # Pattern 4: W[k*n+i, r] = r+1
        W_pattern4 = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W_pattern4[k*n+i, r] = r + 1
        result = _try_fixed_W_decomposition(rank, W_pattern4)
        if result is not None:
            return result
        
        # Pattern 5: W[k*n+i, r] = 1 if (k+i) % 2 == 0 else -1
        W_pattern5 = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W_pattern5[k*n+i, r] = 1 if (k+i) % 2 == 0 else -1
        result = _try_fixed_W_decomposition(rank, W_pattern5)
        if result is not None:
            return result
        
        return None

    # Try to find lower-rank decompositions
    best_rank = rank_schoolbook
    best_result = (U_sb, V_sb, W_sb)
    
    # Try ranks from 12 to rank_schoolbook - 1
    for r in range(12, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        result = _try_systematic_search(r)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If we found a better solution, return it
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        U = [[float(x) for x in row] for row in U]
        V = [[float(x) for x in row] for row in V]
        W = [[float(x) for x in row] for row in W]
        return {"U": U, "V": V, "W": W}
    
    # Otherwise return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
