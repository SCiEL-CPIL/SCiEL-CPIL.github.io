"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 30
    pn = p * n  # 12
    total_eqs = n * m * p  # 60
    
    # The target tensor: T[i,j,k] = 1 if this is a "diagonal" term in the multiplication
    # Actually, the matrix multiplication tensor has T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Wait, let me reconsider. The bilinear form for C = AB where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] has T[i,j,k] = 1 for all valid (i,j,k) in the standard basis
    # No wait - the tensor is defined such that C[i,k] = sum_{j} sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And we want C[i,k] = sum_j A[i,j]*B[j,k]
    # So for the standard basis, the tensor coefficient T[i,j,k] should be 1 for all (i,j,k)
    # because C[i,k] gets contribution A[i,j]*B[j,k] for each j.
    
    # Actually, the tensor T is defined by: for inputs A (n×m) and B (m×p), output C (n×p)
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # In tensor form: C = T(A,B) where T[i,j,k] = 1 for all i,j,k (since every term contributes)
    # Wait no. The tensor T has components T_{i,j,k} such that C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}
    # So T_{i,j,k} = 1 for all valid i,j,k.
    
    # For rank-r decomposition: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try to find low-rank decompositions using a search approach.
    # We'll use a randomized approach with local search.
    
    best_rank = n * m * p  # start with schoolbook
    best_decomp = None
    
    # Generate schoolbook as fallback
    def make_schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    fallback = make_schoolbook()
    
    # Try to find lower rank decompositions
    # Use a structured approach: try ranks from low to high
    
    def verify_decomp(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check all 60 tensor equations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_decomp_numpy(U, V, W, rank):
        """Faster verification using numpy."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k]) * Wa[k*n+i]
                    # That's not right. Let me redo.
                    # s = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # This is element-wise product then sum
                    prod = Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]
                    s = np.sum(prod)
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try a search for lower rank
    # Use random initialization and gradient-free local search
    
    def try_find_decomp(target_rank, max_iters=2000, seed=None):
        """Try to find a rank-r decomposition using random search + refinement."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Random initialization
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Scale to reasonable magnitude
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        # Use a simple iterative approach: for each tensor element, try to fix it
        # This is complex, so let's try a different approach.
        
        # Alternative: use the fact that for small problems, we can set up
        # and solve a system of polynomial equations.
        
        # For now, let's try a heuristic: randomly assign values and check.
        # This is unlikely to work for low ranks, so let's be more systematic.
        
        # Try a structured approach based on known results for small matrix multiplication.
        # For 2×n × n×p, there are known bounds.
        
        # Let's just do a randomized local search
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        err += (s - 1)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                # Check if it's exactly 1 (integer/half-integer)
                U_list = [[int(round(x*2))/2 for x in row] for row in best_U]
                V_list = [[int(round(x*2))/2 for x in row] for row in best_V]
                W_list = [[int(round(x*2))/2 for x in row] for row in best_W]
                if verify_decomp(U_list, V_list, W_list, target_rank):
                    return U_list, V_list, W_list
                # Try rounding to integers
                U_list = [[int(round(x)) for x in row] for row in best_U]
                V_list = [[int(round(x)) for x in row] for row in best_V]
                W_list = [[int(round(x)) for x in row] for row in best_W]
                if verify_decomp(U_list, V_list, W_list, target_rank):
                    return U_list, V_list, W_list
                return None
            
            # Random perturbation
            which = random.choice(['U', 'V', 'W'])
            if which == 'U':
                row = random.randint(0, nm-1)
                col = random.randint(0, target_rank-1)
                U[row, col] += random.gauss(0, 0.1)
            elif which == 'V':
                row = random.randint(0, mp-1)
                col = random.randint(0, target_rank-1)
                V[row, col] += random.gauss(0, 0.1)
            else:
                row = random.randint(0, pn-1)
                col = random.randint(0, target_rank-1)
                W[row, col] += random.gauss(0, 0.1)
        
        return None
    
    # Try ranks from low to high
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also rank >= n+m+p-2 = 2+5+6-2 = 11 (but this is a weak bound)
    # For 2×n × n×p, the rank is at least n + p - 1 + something...
    
    # Let's try a few specific ranks
    for target_rank in [30, 35, 40, 45, 50, 55]:
        if time.time() > deadline - 10:
            break
        
        result = try_find_decomp(target_rank, max_iters=1000, seed=target_rank)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = target_rank
            best_decomp = {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
            break
    
    # If we found something better, use it; otherwise use schoolbook
    if best_decomp is not None:
        return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
    else:
        return {"U": fallback["U"], "V": fallback["V"], "W": fallback["W"]}
# EVOLVE-BLOCK-END
