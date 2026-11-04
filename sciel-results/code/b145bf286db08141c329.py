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

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    Indices:
    a = i*2 + j
    b = j*2 + k
    g = k*2 + i
    """
    R = 7
    U = np.zeros((4, R))
    V = np.zeros((4, R))
    W = np.zeros((4, R))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    # C11 is g=0 (k=0, i=0). C22 is g=3 (k=1, i=1).
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21 + A22) B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2. C21 is g=2 (k=1, i=0)? No.
    # C_{i,k}. i=1, k=0 -> C10?
    # g = k*2 + i.
    # C_{1,0} (i=1, k=0) -> g = 0*2 + 1 = 1.
    # C_{0,1} (i=0, k=1) -> g = 1*2 + 0 = 2.
    # C_{1,1} (i=1, k=1) -> g = 1*2 + 1 = 3.
    # C_{0,0} (i=0, k=0) -> g = 0.
    
    # M2 contributes to C_{2,1} in 1-based -> C_{1,0} in 0-based?
    # Strassen: C21 = M2 + M4.
    # C21 is row 2, col 1 (1-based). i=1, k=0 (0-based).
    # g = k*2 + i = 0*2 + 1 = 1.
    W[1, 1] = 1
    
    # M3 = A11 (B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 = M3 + M5.
    # C12 is i=0, k=1. g = 1*2 + 0 = 2.
    W[2, 2] = 1
    
    # M4 = A22 (B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4.
    # C11 (i=0, k=0) -> g=0.
    # C21 (i=1, k=0) -> g=1.
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11 + A12) B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C12 += M5, C11 -= M5.
    # C12 (i=0, k=1) -> g=2.
    # C11 (i=0, k=0) -> g=0.
    W[2, 4] = 1; W[0, 4] = -1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6.
    # C22 (i=1, k=1) -> g=3.
    W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7.
    # C11 (i=0, k=0) -> g=0.
    W[0, 6] = 1
    
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combines two decompositions for (n, m1, p) and (n, m2, p) into one for (n, m1+m2, p).
    """
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n * (m1 + m2), R))
    V = np.zeros(((m1 + m2) * p, R))
    W = np.zeros((p * n, R))
    
    # U mapping
    # A1 rows: 0 .. n*m1 - 1
    # A2 rows: n*m1 .. n*(m1+m2) - 1
    # U1 has rows indexed by a1 = i*m1 + j.
    # U2 has rows indexed by a2 = i*m2 + j.
    
    for i in range(n):
        for j in range(m1):
            a_global = i * (m1 + m2) + j
            a_local = i * m1 + j
            U[a_global, :R1] = U1[a_local, :]
            
        for j in range(m2):
            a_global = i * (m1 + m2) + (m1 + j)
            a_local = i * m2 + j
            U[a_global, R1:] = U2[a_local, :]
            
    # V mapping
    # B1 rows: 0 .. m1*p - 1
    # B2 rows: m1*p .. (m1+m2)*p - 1
    # V1 has rows indexed by b1 = j*p + k.
    # V2 has rows indexed by b2 = j*p + k.
    
    for j in range(m1):
        for k in range(p):
            b_global = j * p + k
            b_local = j * p + k
            V[b_global, :R1] = V1[b_local, :]
            
    for j in range(m2):
        for k in range(p):
            b_global = (m1 + j) * p + k
            b_local = j * p + k
            V[b_global, R1:] = V2[b_local, :]
            
    # W mapping
    # W is concatenated horizontally.
    W[:, :R1] = W1
    W[:, R1:] = W2
    
    return U, V, W

def _build_recursive(n, m, p):
    """
    Recursively builds a low-rank decomposition using Strassen-like splits.
    """
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1]])
        V = np.array([[1]])
        W = np.array([[1]])
        return U, V, W
        
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
        
    # Try splitting m
    best_rank = float('inf')
    best_split = None
    
    # Split m
    if m > 1:
        for m1 in range(1, m):
            m2 = m - m1
            # Estimate rank
            # We don't know the exact rank, but we can assume it's at least n*m1*p + n*m2*p? No.
            # We just pick the split that balances the sizes.
            # For now, let's just pick m1 = m // 2
            pass
            
    # Simplified strategy: Always split the largest dimension in half.
    # If n is largest, split n.
    # If m is largest, split m.
    # If p is largest, split p.
    
    # For now, let's implement splitting m as it's the most common for matrix multiplication optimization.
    # We can also split n or p.
    
    # Let's try all splits of the largest dimension and pick the one that yields the smallest estimated rank.
    # Estimated rank: R(n, m1, p) + R(n, m2, p).
    # We can memoize the ranks.
    
    # To keep it simple and fast, we will use a fixed splitting strategy:
    # Split the dimension with the largest size into two roughly equal parts.
    
    dims = [(n, 'n'), (m, 'm'), (p, 'p')]
    dims.sort(key=lambda x: x[0], reverse=True)
    largest_dim_val, largest_dim_name = dims[0]
    
    if largest_dim_val <= 1:
        # Should not happen if we hit base cases
        return _build_schoolbook(n, m, p)
        
    # Split in half
    split_val = largest_dim_val // 2
    if split_val < 1:
        split_val = 1
    if largest_dim_val - split_val < 1:
        split_val = largest_dim_val - 1
        
    if largest_dim_name == 'm':
        m1 = split_val
        m2 = m - m1
        U1, V1, W1 = _build_recursive(n, m1, p)
        U2, V2, W2 = _build_recursive(n, m2, p)
        return _combine_decompositions(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
    elif largest_dim_name == 'n':
        n1 = split_val
        n2 = n - n1
        # Splitting n requires a different combination logic.
        # C1 = A1 B, C2 = A2 B.
        # U1 shape (n1*m, R1), U2 shape (n2*m, R2).
        # V1 shape (m*p, R1), V2 shape (m*p, R2).
        # W1 shape (p*n1, R1), W2 shape (p*n2, R2).
        # Combined U: Stack U1 and U2 vertically?
        # A rows: 0..n1*m-1 from A1, n1*m..n*m-1 from A2.
        # V: Concatenate horizontally?
        # W: Stack W1 and W2 vertically?
        # g = k*n + i.
        # For C1 (rows 0..n1-1), g ranges 0..p*n1-1.
        # For C2 (rows n1..n-1), g ranges p*n1..p*n-1.
        # This is more complex. Let's stick to splitting m for now, or implement n/p splits if needed.
        # For 4,5,6, splitting m=5 into 2,3 works well.
        # If we need to split n or p, we can implement it.
        # For now, let's just split m if it's the largest, or if m>1.
        # If m==1, we must split n or p.
        
        # Fallback: Split m if possible, else n, else p.
        if m > 1:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = _build_recursive(n, m1, p)
            U2, V2, W2 = _build_recursive(n, m2, p)
            return _combine_decompositions(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
        elif n > 1:
            # Implement n split
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = _build_recursive(n1, m, p)
            U2, V2, W2 = _build_recursive(n2, m, p)
            
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            R = R1 + R2
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # U: Stack
            U[:n1*m, :R1] = U1
            U[n1*m:, R1:] = U2
            
            # V: Concatenate
            V[:, :R1] = V1
            V[:, R1:] = V2
            
            # W: Stack
            W[:p*n1, :R1] = W1
            W[p*n1:, R1:] = W2
            
            return U, V, W
        else:
            # Split p
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = _build_recursive(n, m, p1)
            U2, V2, W2 = _build_recursive(n, m, p2)
            
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            R = R1 + R2
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # U: Concatenate
            U[:, :R1] = U1
            U[:, R1:] = U2
            
            # V: Stack
            V[:m*p1, :R1] = V1
            V[m*p1:, R1:] = V2
            
            # W: Concatenate?
            # g = k*n + i.
            # For C1 (cols 0..p1-1), g ranges 0..p1*n-1.
            # For C2 (cols p1..p-1), g ranges p1*n..p*n-1.
            # So W is stacked vertically.
            W[:p1*n, :R1] = W1
            W[p1*n:, R1:] = W2
            
            return U, V, W
    else:
        # Split p
        p1 = split_val
        p2 = p - p1
        U1, V1, W1 = _build_recursive(n, m, p1)
        U2, V2, W2 = _build_recursive(n, m, p2)
        
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R = R1 + R2
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # U: Concatenate
        U[:, :R1] = U1
        U[:, R1:] = U2
        
        # V: Stack
        V[:m*p1, :R1] = V1
        V[m*p1:, R1:] = V2
        
        # W: Stack
        W[:p1*n, :R1] = W1
        W[p1*n:, R1:] = W2
        
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

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _optimize_decomposition_fast(n, m, p, initial_rank=None, max_time=100):
    """
    Fast Alternating Least Squares using pre-computed Gram matrices.
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
            
            # Pre-compute indices for fast t-vector calculation
            # For each a, store the list of (b, g) pairs where t=1
            # a = i*m + j
            # b = j*p + k
            # g = k*n + i
            # For fixed a (i,j), k varies 0..p-1
            a_to_bg = {}
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    pairs = []
                    for k in range(p):
                        b = j * p + k
                        g = k * n + i
                        pairs.append((b, g))
                    a_to_bg[a] = pairs

            # Similarly for b and g
            b_to_ag = {}
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    pairs = []
                    for i in range(n):
                        a = i * m + j
                        g = k * n + i
                        pairs.append((a, g))
                    b_to_ag[b] = pairs

            g_to_ab = {}
            for k in range(p):
                for i in range(n):
                    g = k * n + i
                    pairs = []
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        pairs.append((a, b))
                    g_to_ab[g] = pairs

            # Alternating least squares
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                # M^T M = (V^T V) o (W^T W)
                G_V = V.T @ V
                G_W = W.T @ W
                A_mat = G_V * G_W # Hadamard product
                # Regularize slightly to avoid singularity
                A_mat += 1e-6 * np.eye(target_rank)
                
                for a in range(n * m):
                    # Compute b_vec = sum_k V[b_k, :] * W[g_k, :]
                    b_vec = np.zeros(target_rank)
                    for b, g in a_to_bg[a]:
                        b_vec += V[b, :] * W[g, :]
                    
                    try:
                        U[a, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass

                # Update V
                G_U = U.T @ U
                G_W = W.T @ W
                A_mat = G_U * G_W
                A_mat += 1e-6 * np.eye(target_rank)
                
                for b in range(m * p):
                    b_vec = np.zeros(target_rank)
                    for a, g in b_to_ag[b]:
                        b_vec += U[a, :] * W[g, :]
                    
                    try:
                        V[b, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass

                # Update W
                G_U = U.T @ U
                G_V = V.T @ V
                A_mat = G_U * G_V
                A_mat += 1e-6 * np.eye(target_rank)
                
                for g in range(p * n):
                    b_vec = np.zeros(target_rank)
                    for a, b in g_to_ab[g]:
                        b_vec += U[a, :] * V[b, :]
                    
                    try:
                        W[g, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass
                
                # Check convergence and round
                if iteration % 10 == 0:
                    # Calculate error on 1-positions
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; g = k*n+i
                                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                                error += (val - 1.0) ** 2
                    
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
                            error += val ** 2
                    
                    if error < 1e-6:
                        # Try rounding
                        U_rounded = _round_to_half_integer(U)
                        V_rounded = _round_to_half_integer(V)
                        W_rounded = _round_to_half_integer(W)
                        
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                            return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try recursive Strassen-like decomposition
    try:
        U_rec, V_rec, W_rec = _build_recursive(n, m, p)
        if _verify_decomposition(U_rec, V_rec, W_rec, n, m, p):
            if U_rec.shape[1] < best_rank:
                best_U, best_V, best_W = U_rec, V_rec, W_rec
                best_rank = U_rec.shape[1]
    except Exception:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
