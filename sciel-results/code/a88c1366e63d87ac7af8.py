"""Matrix multiplication 4x5x6 - Strassen-style decomposition search."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact (within tolerance)."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Build the tensor: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Check against schoolbook: T[i,j,k] should be 1 if (i,j,k) is a valid triple
    # Actually, the tensor for matmul is: T[i,j,k] = delta (identity in appropriate sense)
    # The matrix multiplication tensor: M[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
    # Wait, let me reconsider. The matmul tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Efficient check: for each c, the outer product U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # We need the sum to equal the all-ones tensor of shape (n*m, m*p, p*n) restricted to valid indices
    
    # Build full tensor
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # T_full = sum_c U[:,c] * V[:,c] * W[:,c] (outer product summed)
    # Use einsum for efficiency
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    # The target: for valid (i,j,k) where i<m, j<p (wait, need to be careful with indexing)
    # U row index: i*m + j where i in [0,n), j in [0,m)
    # V row index: j*p + k where j in [0,m), k in [0,p)
    # W row index: k*n + i where k in [0,p), i in [0,n)
    # So T[i*m+j, j*p+k, k*n+i] should = 1 for all valid i,j,k
    
    # Create target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    diff = np.abs(T - target)
    return np.max(diff) < 1e-9

def _strassen_decompose(n, m, p):
    """
    Try to build a Strassen-style decomposition for n x m * m x p.
    Returns (U, V, W) or None if it can't be done cleanly.
    
    Strategy: Split the problem into sub-problems and use 7 multiplications
    for 2x2x2 blocks where possible.
    """
    if n == 1 and m == 1 and p == 1:
        U = [[1.0]]
        V = [[1.0]]
        W = [[1.0]]
        return U, V, W
    
    if n == 2 and m == 2 and p == 2:
        # Strassen 7-term decomposition for 2x2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = AB, C[0,0]=ae+bg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh
        # Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C[0,0] = M1+M4-M5+M7
        # C[0,1] = M3+M5
        # C[1,0] = M2+M4
        # C[1,1] = M1-M2+M3+M6
        
        # Indexing: i in {0,1}, j in {0,1}, k in {0,1}
        # U row: i*2+j, V row: j*2+k, W row: k*2+i
        # A elements: A[0,0]=a (i=0,j=0), A[0,1]=b (i=0,j=1), A[1,0]=c (i=1,j=0), A[1,1]=d (i=1,j=1)
        # B elements: B[0,0]=e (j=0,k=0), B[0,1]=f (j=0,k=1), B[1,0]=g (j=1,k=0), B[1,1]=h (j=1,k=1)
        # C elements: C[0,0] (i=0,k=0), C[0,1] (i=0,k=1), C[1,0] (i=1,k=0), C[1,1] (i=1,k=1)
        
        # For each Strassen term M_t, we need U[*,t], V[*,t], W[*,t]
        # M1 = (a+d)(e+h): U has 1 at (0,0) and (1,1) [i*m+j: 0*2+0=0, 1*2+1=3]
        #                                  V has 1 at (0,0) and (1,1) [j*p+k: 0*2+0=0, 1*2+1=3]
        #                                  W: C[0,0] gets +1, C[1,1] gets +1
        #                                  W row k*2+i: C[0,0] is k=0,i=0 -> 0; C[1,1] is k=1,i=1 -> 3
        
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (a+d)(e+h)
        U[0,0] = 1; U[3,0] = 1  # a, d
        V[0,0] = 1; V[3,0] = 1  # e, h
        W[0,0] = 1; W[3,0] = 1  # C[0,0], C[1,1]
        
        # M2 = (c+d)e
        U[2,1] = 1; U[3,1] = 1  # c, d
        V[0,1] = 1              # e
        W[2,1] = 1; W[3,1] = -1 # C[1,0], -C[1,1]
        
        # M3 = a(f-h)
        U[0,2] = 1              # a
        V[1,2] = 1; V[3,2] = -1 # f, -h
        W[1,2] = 1; W[3,2] = 1  # C[0,1], C[1,1]
        
        # M4 = d(g-e)
        U[3,3] = 1              # d
        V[0,3] = -1; V[2,3] = 1 # -e, g
        W[0,3] = 1; W[2,3] = 1  # C[0,0], C[1,0]
        
        # M5 = (a+b)h
        U[0,4] = 1; U[1,4] = 1  # a, b
        V[3,4] = 1              # h
        W[0,4] = -1; W[1,4] = 1 # -C[0,0], C[0,1]
        
        # M6 = (c-a)(e+f)
        U[0,5] = -1; U[2,5] = 1 # -a, c
        V[0,5] = 1; V[1,5] = 1  # e, f
        W[3,5] = 1              # C[1,1]
        
        # M7 = (b-d)(g+h)
        U[1,6] = 1; U[3,6] = -1 # b, -d
        V[2,6] = 1; V[3,6] = 1  # g, h
        W[0,6] = 1; W[1,6] = -1 # C[0,0], -C[0,1]
        
        return U.tolist(), V.tolist(), W.tolist()
    
    # General case: try to split into 2x2 blocks where possible
    # For 4x5x6: split A into 2x2 blocks (vertically: 4=2*2, horizontally: 5... doesn't split evenly)
    # This is tricky. Let's try a different approach.
    
    return None

def _build_from_strassen_blocks(n, m, p):
    """
    Build a decomposition by splitting into Strassen-able sub-blocks.
    For 4x5x6: 
    - A is 4x5: split into top 2x5 and bottom 2x5
    - B is 5x6: split into left 5x3 and right 5x3
    - C is 4x6: 2x2 blocks of 2x3
    
    Actually, let's think of it as:
    C = A*B where A is 4x5, B is 5x6.
    Split A into A1 (2x5) and A2 (2x5)
    Split B into B1 (5x3) and B2 (5x3)
    C11 = A1*B1 (2x3), C12 = A1*B2 (2x3)
    C21 = A2*B1 (2x3), C22 = A2*B2 (2x3)
    
    Each 2x3 sub-problem can use a 2x2x2 Strassen plus extra terms.
    """
    # For 2x3: A is 2x3, B is 3x3, C is 2x3
    # Split B into B_left (3x2) and B_right (3x1)
    # C_left = A * B_left (2x2) -> Strassen 7 terms
    # C_right = A * B_right (2x1) -> 2 terms (schoolbook)
    # Total: 9 terms per 2x3 block
    
    # 4 blocks * 9 = 36 terms. That's worse than 23.
    # Need a better approach.
    
    # Alternative: use the known R(2,3,3) <= 23 result.
    # This is a known result from algebraic complexity theory.
    # The decomposition exists but is non-trivial.
    
    # Let me try to construct it using the "laser method" or known constructions.
    # Actually, for a programming task, let me try a numerical search approach.
    
    return None

def _numerical_search(n, m, p, max_rank, time_limit):
    """
    Try to find a low-rank decomposition using numerical optimization.
    Start from a random or structured initialization and optimize.
    """
    start_time = time.time()
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But we know R(4,5,6) = R(2,3,3) <= 23
    # So try ranks 23, 24, ..., up to some limit
    
    for rank in range(23, min(max_rank + 1, n*m*p + 1)):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            np.random.seed(42 + rank * 100 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Optimize: minimize sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            # This is a non-convex problem. Use simple gradient descent or alternating minimization.
            
            # Alternating minimization: fix two, solve for third (linear least squares)
            for iteration in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U: for each (i,j), U[i*m+j,:] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is: U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:] elementwise) = 1 for all k
                # Let's build the linear system for each (i,j)
                
                # Actually, let's use a simpler approach: build the full linear system
                # For fixed V and W, the problem for U is linear.
                # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                # = U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:])
                
                # For each row of U (index r = i*m+j), we need:
                # For all k: sum_c U[r,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is a system of p equations in rank unknowns (for each row r)
                
                # Build matrix A_r where A_r[k,c] = V[j*p+k,c] * W[k*n+i,c]
                # Then A_r @ U[r,:] = ones(p)
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        r = i * m + j
                        # A_r is p x rank
                        A_r = np.zeros((p, rank))
                        for k in range(p):
                            A_r[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A_r @ U[r,:] = ones(p)
                        try:
                            U_new[r, :] = np.linalg.lstsq(A_r, np.ones(p), rcond=None)[0]
                        except:
                            U_new[r, :] = U[r, :]
                U = U_new
                
                # Similarly for V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        r = j * p + k
                        # For fixed U and W, need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                        # V[r,:] dot (U[i*m+j,:] * W[k*n+i,:]) = 1 for all i
                        A_r = np.zeros((n, rank))
                        for i in range(n):
                            A_r[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V_new[r, :] = np.linalg.lstsq(A_r, np.ones(n), rcond=None)[0]
                        except:
                            V_new[r, :] = V[r, :]
                V = V_new
                
                # Similarly for W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        r = k * n + i
                        # For fixed U and V, need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                        # W[r,:] dot (U[i*m+j,:] * V[j*p+k,:]) = 1 for all j
                        A_r = np.zeros((m, rank))
                        for j in range(m):
                            A_r[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W_new[r, :] = np.linalg.lstsq(A_r, np.ones(m), rcond=None)[0]
                        except:
                            W_new[r, :] = W[r, :]
                W = W_new
                
                # Check convergence
                T = np.einsum('ic,jc,kc->ijk', U, V, W)
                target = np.zeros((n*m, m*p, p*n))
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            target[ii*m+jj, jj*p+kk, kk*n+ii] = 1.0
                err = np.max(np.abs(T - target))
                if err < 1e-8:
                    return U.tolist(), V.tolist(), W.tolist(), rank
        
        # Check if this rank works
        if time.time() - start_time > time_limit:
            break
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, try the known Strassen-based construction for R(2,3,3) <= 23
    # This requires a specific algebraic construction.
    # Let me try to build it step by step.
    
    # For now, use the schoolbook as fallback
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
    
    # Try to find a better decomposition
    result = _numerical_search(n, m, p, 40, 130)
    
    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
