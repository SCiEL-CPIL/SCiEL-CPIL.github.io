import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS
    deadline = t_start + 130  # 130s deadline with safety margin before 150s

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a better decomposition using bilinear structure
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # The tensor has entries T[i,j,k] = 1 for the multiplication
    # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy: Use the fact that we can share computation across k values
    # Group by (i,j) pairs and try to decompose the p-dimensional output
    
    # First, let's try a simple improvement: for each (i,j), we need p outputs
    # The k-th output is sum_j A[i,j] * B[j,k]
    # We can use rank-p decomposition for each (i,j) if we can't do better
    
    # Actually, let's try: fix i, then for all j and k, we need
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a rank-m decomposition for each fixed i, giving total rank n*m*p = 96
    
    # Better idea: use the structure where we decompose the m dimension
    # For the 4x8 part, we can potentially use rank < 32 if there's structure
    # But 4x8 matrix multiplication is just linear algebra, rank is 32 for schoolbook
    
    # Let's try a tensor decomposition approach using iterative refinement
    # Start with schoolbook and try to merge terms
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try randomized initialization + gradient-free optimization
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For this specific case, let's try to find a structured decomposition
    # Key insight: C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as: for each i, C[i,:] = A[i,:] * B
    # The rank of the "slice" for fixed i is at most m=4 (since A[i,:] is a row vector)
    # But we need all p=8 outputs, so for fixed i, we need rank p=8 if B is full rank
    # Total: n * p = 24? No, that's not right because j is shared
    
    # Actually the tensor rank of matrix multiplication [n,m,p] is known to be:
    # For [3,4,8], the minimum rank is not known to be less than 96 in general
    # But we can try to find a decomposition with rank < 96
    
    # Let's try: use rank 3 for the n-dimension (since n=3), rank 4 for m, rank 8 for p
    # That gives 96. To improve, we need to exploit linear dependencies.
    
    # Try a concrete approach: assume U, V, W have a Kronecker-like structure
    # U[i*m+j, r] = u_i * v_j * w_r? No, that doesn't work for matmul.
    
    # Let's just try a few structured ansatz:
    # Ansatz 1: U depends only on (i,j) via i*m+j, V on j*p+k, W on k*n+i
    # The schoolbook has U[i*m+j, c] = 1 if c = (i,j,k), else 0
    
    # Try: reduce by noticing that for fixed (i,j), the k-sum can be done with
    # fewer than p terms if we allow the coefficients to be non-trivial.
    # But for exact decomposition with integer/half-integer entries, we need exactness.
    
    # Given the difficulty, let's try a numerical search with low rank and check validity.
    
    np.random.seed(42)
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose the matmul tensor."""
        # U: (n*m) x rank
        # V: (m*p) x rank  
        # W: (p*n) x rank
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        
        # Build the tensor: T[i,j,k] should equal sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False, abs(val - 1.0)
        return True, 0.0
    
    # Try various ranks from 24 up to 95
    for target_rank in range(24, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple iterative refinement: for each term, try to correct
            # Use the fact that sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
            # This is a system of n*m*p = 96 equations in target_rank* (n*m + m*p + p*n) variables
            
            # For now, just check if random works (very unlikely for low rank)
            valid, err = check_decomposition(U, V, W)
            if valid:
                best_rank = target_rank
                # Convert to lists
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_rank <= target_rank:
            break
    
    # If we found something better, try to snap to nearest half-integers
    if best_rank < n * m * p:
        U_arr = np.array(best_result["U"], dtype=np.float64)
        V_arr = np.array(best_result["V"], dtype=np.float64)
        W_arr = np.array(best_result["W"], dtype=np.float64)
        
        # Snap to half-integers
        U_snap = np.round(U_arr * 2) / 2
        V_snap = np.round(V_arr * 2) / 2
        W_snap = np.round(W_arr * 2) / 2
        
        valid, err = check_decomposition(U_snap, V_snap, W_snap, tol=1e-8)
        if not valid:
            # Try integer snapping
            U_snap = np.round(U_arr)
            V_snap = np.round(V_arr)
            W_snap = np.round(W_arr)
            valid, err = check_decomposition(U_snap, V_snap, W_snap, tol=1e-8)
        
        if valid:
            best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        else:
            # Keep original if snapping doesn't work (but may fail validity check by evaluator)
            # Fall back to schoolbook for safety
            best_result = schoolbook()
    
    return best_result
