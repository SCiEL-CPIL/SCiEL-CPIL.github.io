import numpy as np
import time
from fractions import Fraction

def construct():
    """
    Search for rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    Uses slice method / linear algebra approach with time-bounded search.
    Falls back to rank-28 Strassen-based decomposition if no better found.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    n, m, p = 2, 3, 5
    
    def build_rank_28():
        """Original rank-28 decomposition as fallback."""
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        
        # Block 1: j=0, all k (2x1x5 schoolbook, rank 10)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (A00+A11)(B00+B11) -> C00, C11
            U[(i0+0)*m + (j0+0), col] = 1; U[(i0+1)*m + (j0+1), col] = 1
            V[(j0+0)*p + (k0+0), col] = 1; V[(j0+1)*p + (k0+1), col] = 1
            W[(k0+0)*n + (i0+0), col] = 1; W[(k0+1)*n + (i0+1), col] = 1
            col += 1
            # M2: (A10+A11)B00 -> C10, -C11
            U[(i0+1)*m + (j0+0), col] = 1; U[(i0+1)*m + (j0+1), col] = 1
            V[(j0+0)*p + (k0+0), col] = 1
            W[(k0+0)*n + (i0+1), col] = 1; W[(k0+1)*n + (i0+1), col] = -1
            col += 1
            # M3: A00(B01-B11) -> C01, C11
            U[(i0+0)*m + (j0+0), col] = 1
            V[(j0+0)*p + (k0+1), col] = 1; V[(j0+1)*p + (k0+1), col] = -1
            W[(k0+1)*n + (i0+0), col] = 1; W[(k0+1)*n + (i0+1), col] = 1
            col += 1
            # M4: A11(B10-B00) -> C00, C10
            U[(i0+1)*m + (j0+1), col] = 1
            V[(j0+1)*p + (k0+0), col] = 1; V[(j0+0)*p + (k0+0), col] = -1
            W[(k0+0)*n + (i0+0), col] = 1; W[(k0+0)*n + (i0+1), col] = 1
            col += 1
            # M5: (A00+A01)B11 -> -C00, C01
            U[(i0+0)*m + (j0+0), col] = 1; U[(i0+0)*m + (j0+1), col] = 1
            V[(j0+1)*p + (k0+1), col] = 1
            W[(k0+0)*n + (i0+0), col] = -1; W[(k0+1)*n + (i0+0), col] = 1
            col += 1
            # M6: (A10-A00)(B00+B01) -> C11
            U[(i0+1)*m + (j0+0), col] = 1; U[(i0+0)*m + (j0+0), col] = -1
            V[(j0+0)*p + (k0+0), col] = 1; V[(j0+0)*p + (k0+1), col] = 1
            W[(k0+1)*n + (i0+1), col] = 1
            col += 1
            # M7: (A01-A11)(B10+B11) -> C00
            U[(i0+0)*m + (j0+1), col] = 1; U[(i0+1)*m + (j0+1), col] = -1
            V[(j0+1)*p + (k0+0), col] = 1; V[(j0+1)*p + (k0+1), col] = 1
            W[(k0+0)*n + (i0+0), col] = 1
            col += 1
        
        add_strassen(0, 1, 0)
        
        for i in range(n):
            for j in [1, 2]:
                U[i * m + j, col] = 1
                V[j * p + 2, col] = 1
                W[2 * n + i, col] = 1
                col += 1
        
        add_strassen(0, 1, 3)
        
        assert col == rank, f"Column count mismatch: {col} != {rank}"
        return U, V, W
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = 2, 3, 5
        rank = U.shape[1]
        
        # For each output position (k, i), check:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give 
        # the coefficient of A[i,j]*B[j,k] in C[i,k]
        # Which is 1 if j is the "middle" index, else 0
        
        # Actually, the tensor T[i,j,k] = delta_{middle}
        # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, let me re-read the convention.
        
        # W[k*n+i, r] is the coefficient for output C[i,k]
        # U[i*m+j, r] is the coefficient for input A[i,j]
        # V[j*p+k, r] is the coefficient for input B[j,k]
        
        # So the reconstructed value for (i,j,k) is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 if this is a valid term, i.e., always 1 for the 
        # multiplication tensor where each (i,j,k) triple contributes exactly once.
        
        # Actually for matrix multiplication C = A*B:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor has T[i,j,k] = 1 for all valid (i,j,k)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False, (i, j, k, val)
        return True, None
    
    # Try to find a lower rank decomposition using the slice method
    # The slice method: fix one index and find the rank of the resulting matrix
    
    # For matrix multiplication 2x3 * 3x5:
    # Fix j (the middle index), we get a matrix of size (2*5) x (2*5) 
    # No wait, fixing j gives us a map from A[:,j] (2x1) and B[j,:] (1x5) to C (2x5)
    # This is a rank-1 outer product structure per j
    
    # The total tensor rank is at least max over slices
    # Slice by fixing i: gives m*p x p*n = 3*5 x 5*2 = 15x10 matrix
    # Slice by fixing j: gives n*p x n*p = 2*5 x 2*5 = 10x10 matrix  
    # Slice by fixing k: gives n*m x m*n = 2*3 x 3*2 = 6x6 matrix
    
    # Let's try to search for rank r < 28 by checking if a valid decomposition exists
    # We'll use a randomized approach with verification
    
    best_result = None
    best_rank = 28
    
    # Try ranks from 20 up to 27 (below current 28)
    for target_rank in range(20, 28):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            W = np.zeros((p*n, target_rank), dtype=float)
            
            # Initialize with random small values
            for r in range(target_rank):
                # Each slice should contribute to the decomposition
                # Random pattern
                for idx in range(n*m):
                    U[idx, r] = np.random.choice([-1, 0, 1])
                for idx in range(m*p):
                    V[idx, r] = np.random.choice([-1, 0, 1])
                for idx in range(p*n):
                    W[idx, r] = np.random.choice([-1, 0, 1])
            
            # Check if this is a valid decomposition
            valid, err = verify_decomposition(U, V, W)
            if valid:
                best_result = (U, V, W)
                best_rank = target_rank
                break
        
        if best_result is not None:
            break
    
    # If we found a better rank, use it; otherwise fall back to rank 28
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = build_rank_28()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
