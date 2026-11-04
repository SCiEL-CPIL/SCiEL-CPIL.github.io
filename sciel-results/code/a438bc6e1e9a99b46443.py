"""Matrix multiplication 2x4x4 rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    Output factor follows W[k*n+i, r] convention.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    # The multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We need U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r to give 1 for all (i,j,k)
    # Actually the tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] such that C[i,k] = sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Wait, let me reconsider the convention.
    
    # Standard matrix multiplication tensor:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T has indices (i, j, k) where T[i,j,k] = 1 (it's the structure tensor)
    # Rank-r decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # U has shape (n*m, rank) indexed by (i*m+j)
    # V has shape (m*p, rank) indexed by (j*p+k)  
    # W has shape (p*n, rank) indexed by (k*n+i)
    
    # The schoolbook decomposition has rank = n*m*p = 32
    # For 2x4x4, we can potentially do better.
    
    # Known results: 
    # - Strassen's algorithm gives rank 7 for 2x2x2 (vs 8 schoolbook)
    # - For 2x4x4, we can think of it as multiplying a 2x4 by 4x4
    # - This can be viewed as 4 separate 2x1x4 multiplications or other groupings
    
    # Let's use a systematic approach: try to find a lower rank decomposition
    # using numerical optimization with exact rational recovery.
    
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def check_time():
        return elapsed() < deadline
    
    # First, let's try a structured approach.
    # For 2x4x4, we can partition the 4x4 matrix into blocks.
    # A is 2x4, B is 4x4, C is 2x4.
    
    # One approach: use Strassen-like ideas.
    # Think of A as [A1 | A2] where A1, A2 are 2x2.
    # Think of B as [[B11, B12], [B21, B22]] where each Bij is 2x2.
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    # Each 2x2*2x2 multiplication can be done with rank 7 (Strassen).
    # But we have cross terms that complicate things.
    
    # Alternative: treat this as a bilinear map from R^{8} x R^{16} -> R^{8}
    # and use numerical rank minimization.
    
    # Let's try a direct numerical approach with random initialization
    # and gradient-free optimization, then recover exact rationals.
    
    def try_decomposition(rank, max_iter=5000, seed=42):
        """Try to find a rank-r decomposition using numerical optimization."""
        if not check_time():
            return None
        
        rng = np.random.default_rng(seed)
        
        # Initialize U, V, W randomly
        U = rng.standard_normal((nm, rank))
        V = rng.standard_normal((mp, rank))
        W = rng.standard_normal((pn, rank))
        
        # The tensor T[i,j,k] = 1 for all i,j,k
        # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        
        # This is a system of n*m*p = 32 equations in rank*32 variables (but with structure)
        # Actually it's rank*(nm + mp + pn) = rank*32 variables total, but we need
        # the product to match the tensor.
        
        # Let's set up the problem differently. Fix U and V, solve for W, etc.
        # Or use alternating least squares.
        
        # Target tensor values
        target = np.ones((n, m, p))
        
        # Current residual
        def compute_residual(U, V, W):
            """Compute the tensor from the decomposition."""
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            T = np.zeros((n, m, p))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            return T
        
        def residual_norm(U, V, W):
            T = compute_residual(U, V, W)
            return np.linalg.norm(T - target)
        
        # Alternating least squares
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if not check_time():
                break
            
            # Update W: for each (k,i), we have a linear system
            # sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] should contribute to target
            # Actually, for fixed (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # For fixed (k,i), varying j: this is a system in W[k*n+i, :]
            
            # Let's do proper ALS.
            # Fix U and V, solve for W.
            # For each (k,i): sum_r (sum_j U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = sum_j target[i,j,k]
            # Wait, that's not right either. The equation is:
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # 
            # If we fix U and V, then for each (k,i), we have m equations (one for each j):
            # sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1 for j=0,...,m-1
            # This is a linear system: M_{ki} @ W[k*n+i, :] = 1
            # where M_{ki}[j, r] = U[i*m+j,r] * V[j*p+k,r]
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Fix V and W, solve for U.
            # For each (i,j): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for each k
            # So for fixed (i,j): M2 @ U[i*m+j, :] = 1 where M2[k, r] = V[j*p+k,r]*W[k*n+i,r]
            for i in range(n):
                for j in range(m):
                    M2 = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            M2[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M2, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Fix U and W, solve for V.
            # For each (j,k): sum_r (U[i*m+j,r] * W[k*n+i,r]) * V[j*p+k,r] = 1 for each i
            # For fixed (j,k): M3 @ V[j*p+k, :] = 1 where M3[i, r] = U[i*m+j,r]*W[k*n+i,r]
            for j in range(m):
                for k in range(p):
                    M3 = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            M3[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M3, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Compute residual
            res = residual_norm(U, V, W)
            if res < best_residual:
                best_residual = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res < 1e-10:
                break
        
        if best_residual < 1e-8:
            return best_U, best_V, best_W
        return None
    
    def to_exact_fractional(U, V, W, tol=1e-6):
        """Convert numerical solution to exact half-integer/integer entries."""
        from fractions import Fraction
        
        def convert_matrix(M):
            exact = []
            for row in M:
                exact_row = []
                for val in row:
                    # Try to find a half-integer close to val
                    # Half-integers: ..., -1.5, -1, -0.5, 0, 0.5, 1, 1.5, ...
                    best_frac = None
                    best_dist = np.inf
                    # Search in a reasonable range
                    for denom in [1, 2]:
                        for num in range(-100, 101):
                            frac_val = Fraction(num, denom)
                            dist = abs(float(frac_val) - val)
                            if dist < best_dist:
                                best_dist = dist
                                best_frac = frac_val
                    if best_dist < tol:
                        exact_row.append(best_frac)
                    else:
                        # Fall back to nearest half-integer
                        exact_row.append(Fraction(round(val * 2), 2))
                exact.append(exact_row)
            return exact
        
        return convert_matrix(U), convert_matrix(V), convert_matrix(W)
    
    def verify_exact(U_exact, V_exact, W_exact):
        """Verify that the exact decomposition is correct."""
        # Convert to floats for verification
        U_f = np.array([[float(x) for x in row] for row in U_exact])
        V_f = np.array([[float(x) for x in row] for row in V_exact])
        W_f = np.array([[float(x) for x in row] for row in W_exact])
        
        # Check all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(len(U_f[0])):
                        val += U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try ranks from low to high
    # Lower bound: for 2x4x4, the rank is at least max(n*m, m*p, p*n) = 16
    # But likely higher. Let's try from 16 upward.
    
    # Actually, let's first just get a working decomposition, then try to minimize.
    # Start with a known good rank and try to improve.
    
    # Schoolbook: rank 32
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Get schoolbook as fallback
    U_school, V_school, W_school, rank_school = schoolbook()
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find better decompositions
    for rank in range(16, rank_school + 1):
        if not check_time():
            break
        
        # Try multiple seeds
        found = False
        for seed in range(20):
            if not check_time():
                break
            
            result = try_decomposition(rank, max_iter=3000, seed=seed)
            if result is not None:
                U_num, V_num, W_num = result
                U_exact, V_exact, W_exact = to_exact_fractional(U_num, V_num, W_num)
                
                if verify_exact(U_exact, V_exact, W_exact):
                    best_result = (U_exact, V_exact, W_exact)
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Found a decomposition at this rank, try to go lower
            continue
        else:
            # Can't find at this rank, but keep trying higher ranks
            # Actually if we can't find at rank r, we might still find at r+1
            # But let's not waste too much time
            if rank > 24 and not found:
                break
    
    # Convert best result to the required format
    U_best, V_best, W_best = best_result
    
    # Ensure they are in the right format (list of lists with Fraction or int/float)
    # The problem says "integer/half-integer entries"
    
    def finalize(U_mat, V_mat, W_mat):
        """Convert to proper format."""
        U_out = []
        for row in U_mat:
            U_out.append([float(x) if not isinstance(x, Fraction) else float(x) for x in row])
        V_out = []
        for row in V_mat:
            V_out.append([float(x) if not isinstance(x, Fraction) else float(x) for x in row])
        W_out = []
        for row in W_mat:
            W_out.append([float(x) if not isinstance(x, Fraction) else float(x) for x in row])
        return U_out, V_out, W_out
    
    U_final, V_final, W_final = finalize(U_best, V_best, W_best)
    
    return {"U": U_final, "V": V_final, "W": W_final}
