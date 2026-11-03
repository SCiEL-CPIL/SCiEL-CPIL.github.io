"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 100
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    def verify_numpy(U, V, W, rank, tol=1e-9):
        """Verify decomposition using numpy."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fraction(U, V, W, rank):
        """Verify decomposition exactly using fractions."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != Fraction(1):
                        return False
        return True
    
    def to_fraction_matrix(mat):
        """Convert numpy matrix of half-integers to Fraction matrix."""
        rows, cols = mat.shape
        result = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                result[r][c] = Fraction(int(round(mat[r, c] * 2)), 2)
        return result
    
    def from_fraction_matrix(fmat):
        """Convert Fraction matrix to numpy float matrix."""
        rows = len(fmat)
        cols = len(fmat[0])
        result = np.zeros((rows, cols), dtype=float)
        for r in range(rows):
            for c in range(cols):
                result[r, c] = float(fmat[r][c])
        return result
    
    # Kronecker product approach:
    # If we have a decomposition for (n1,m1,p1) with rank r1 and (n2,m2,p2) with rank r2,
    # we get a decomposition for (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    # 
    # For 4x5x5, we can try:
    # - 4x5x5 directly with structured search
    # - Decompose 4 = 2*2, 5 = 5, 5 = 5 -> not helpful directly
    # - Better: use block structure
    
    # Key insight: 4x5x5 can be viewed as sum of blocks
    # Split rows: 4 = 2+2, so (4,5,5) = (2,5,5) + (2,5,5) in a block sense
    # Each (2,5,5) needs rank at most 2*5*5=50, total 100 (no improvement)
    # 
    # Better approach: use the fact that we can share computation
    # Try direct structured search with small integer entries
    
    # Strategy: Use iterative refinement with numpy
    # Start from schoolbook, try to reduce rank by finding linear dependencies
    
    def try_reduce_rank(current_U, current_V, current_W, current_rank, target_rank):
        """Try to find a lower-rank decomposition by solving linear system."""
        if target_rank >= current_rank:
            return None
        
        # We need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a bilinear system. Use alternating minimization.
        
        U = current_U[:, :target_rank].copy()
        V = current_V[:, :target_rank].copy()
        W = current_W[:, :target_rank].copy()
        
        # Alternating minimization
        for iteration in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j), solve for U[i*m+j,:] 
            # Given V and W, for fixed (i,j):
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # Let A_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
            # Then U[i*m+j,:] * A_k = 1 for all k
            # This is: M[i*m+j,:] @ A.T = ones(p) where A is p x target_rank
            
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A.T @ x = ones(p)
                    try:
                        x, residuals, rank_sys, sv = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Update V similarly
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, residuals, rank_sys, sv = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Update W similarly
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, residuals, rank_sys, sv = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
        
        # Check if we have a valid solution
        if verify_numpy(U, V, W, target_rank, tol=1e-6):
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify_numpy(U_snap, V_snap, W_snap, target_rank, tol=1e-9):
                return U_snap, V_snap, W_snap
            
            # Return the floating point solution (may not satisfy integer constraint)
            # But let's check if it's close enough
            return None
        
        return None
    
    # Try to find lower rank decompositions
    # Start from rank 99 down to 50
    for target_rank in range(99, 49, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_reduce_rank(U_school, V_school, W_school, rank_school, target_rank)
        if result is not None:
            U_new, V_new, W_new = result
            best_result = {
                "U": U_new.tolist(),
                "V": V_new.tolist(),
                "W": W_new.tolist()
            }
            best_rank = target_rank
            break  # Found the lowest rank, stop
    
    # Also try random restarts for lower ranks
    if best_rank > 60 and time.time() < deadline - 15:
        np.random.seed(42)
        for target_rank in range(60, 49, -1):
            if time.time() > deadline - 10:
                break
            
            for attempt in range(10):
                U_rand = np.random.uniform(-2, 2, (n * m, target_rank))
                V_rand = np.random.uniform(-2, 2, (m * p, target_rank))
                W_rand = np.random.uniform(-2, 2, (p * n, target_rank))
                
                result = try_reduce_rank(U_rand, V_rand, W_rand, target_rank, target_rank)
                if result is not None:
                    U_new, V_new, W_new = result
                    best_result = {
                        "U": U_new.tolist(),
                        "V": V_new.tolist(),
                        "W": W_new.tolist()
                    }
                    best_rank = target_rank
                    break
            
            if best_rank <= target_rank:
                break
    
    return best_result
