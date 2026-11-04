import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        g = random.randint(0, p*n-1)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[g, r]
            if abs(val) > tol:
                return False
    return True

def _strassen_2x2_block():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    Returns U, V, W factors for the 2x2x2 tensor with rank 7.
    
    For C = AB where A, B are 2x2:
    C[0,0] = M1 - M4 + M6 + M7
    C[0,1] = M5 + M4
    C[1,0] = M1 + M2
    C[1,1] = M3 - M2 + M5 + M7
    
    Where:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))  # 4x7
    V = np.zeros((m*p, rank))  # 4x7
    W = np.zeros((p*n, rank))  # 4x7
    
    # Indexing: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_hybrid(n, m, p):
    """
    Build a hybrid decomposition using Strassen blocks where possible.
    For 4x5x6: partition into blocks and apply Strassen to 2x2 blocks.
    """
    # For 4x5x6, we can use the following strategy:
    # - Use schoolbook for most terms
    # - Apply Strassen optimization to 2x2 sub-blocks
    # 
    # Actually, let's try a different approach: use the known result that
    # rank(M_{4,5,6}) <= some value from known algorithms.
    # 
    # For now, let's try to find a good decomposition through optimization.
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return U, V, W

def _optimize_decomposition(n, m, p, initial_rank=None, max_time=100):
    """
    Try to find a lower-rank decomposition using alternating least squares.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    if initial_rank is None:
        initial_rank = n * m * p
    
    # Try decreasing ranks
    for target_rank in range(initial_rank - 1, max(10, initial_rank - 50), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating least squares
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U: for each row a, solve least squares
                # sum_r U[a,r] * V[b,r] * W[g,r] = T[a,b,g]
                # For fixed a: sum_r U[a,r] * (V[:,r] * W[:,r])_elementwise = T[a,*,*]
                
                for a in range(n * m):
                    # Build the design matrix
                    # For each (b,g), the coefficient vector is V[b,:] * W[g,:]
                    # Stack these to form M of shape (m*p*p*n, target_rank)
                    M_rows = []
                    t_vals = []
                    
                    for b in range(m * p):
                        for g in range(p * n):
                            coeff = V[b, :] * W[g, :]
                            M_rows.append(coeff)
                            
                            # Check if (a,b,g) is consistent
                            i = a // m; j = a % m
                            jp = b // p; k = b % p
                            kn = g // n; ip = g % n
                            if i == ip and j == jp and k == kn:
                                t_vals.append(1.0)
                            else:
                                t_vals.append(0.0)
                    
                    M_mat = np.array(M_rows)
                    t_vec = np.array(t_vals)
                    
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                        U[a, :] = sol
                    except:
                        pass
                
                # Update V: for each row b
                for b in range(m * p):
                    M_rows = []
                    t_vals = []
                    
                    for a in range(n * m):
                        for g in range(p * n):
                            coeff = U[a, :] * W[g, :]
                            M_rows.append(coeff)
                            
                            i = a // m; j = a % m
                            jp = b // p; k = b % p
                            kn = g // n; ip = g % n
                            if i == ip and j == jp and k == kn:
                                t_vals.append(1.0)
                            else:
                                t_vals.append(0.0)
                    
                    M_mat = np.array(M_rows)
                    t_vec = np.array(t_vals)
                    
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                        V[b, :] = sol
                    except:
                        pass
                
                # Update W: for each row g
                for g in range(p * n):
                    M_rows = []
                    t_vals = []
                    
                    for a in range(n * m):
                        for b in range(m * p):
                            coeff = U[a, :] * V[b, :]
                            M_rows.append(coeff)
                            
                            i = a // m; j = a % m
                            jp = b // p; k = b % p
                            kn = g // n; ip = g % n
                            if i == ip and j == jp and k == kn:
                                t_vals.append(1.0)
                            else:
                                t_vals.append(0.0)
                    
                    M_mat = np.array(M_rows)
                    t_vec = np.array(t_vals)
                    
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                        W[g, :] = sol
                    except:
                        pass
                
                # Check convergence
                if iteration % 10 == 0:
                    loss = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; g = k*n+i
                                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                                loss += (val - 1.0) ** 2
                    
                    # Sample zero constraints
                    np.random.seed(42)
                    for _ in range(200):
                        a = np.random.randint(0, n*m)
                        b = np.random.randint(0, m*p)
                        g = np.random.randint(0, p*n)
                        i = a // m; j = a % m
                        jp = b // p; k = b % p
                        kn = g // n; ip = g % n
                        if i != ip or j != jp or k != kn:
                            val = np.sum(U[a, :] * V[b, :] * W[g, :])
                            loss += val ** 2
                    
                    if loss < 1e-10:
                        # Verify fully
                        if _verify_decomposition(U, V, W, n, m, p):
                            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try to find a better decomposition
    result = _optimize_decomposition(n, m, p, initial_rank=best_rank, max_time=140)
    
    if result is not None:
        U, V, W = result
        if U.shape[1] < best_rank:
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
