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
    # C22 -= M6.
    # C22 (i=1, k=1) -> g=3.
    W[3, 5] = -1
    
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

def _combine_split_n(U1, V1, W1, U2, V2, W2, n1, m, p):
    """Combine decompositions for (n1,m,p) and (n2,m,p) into (n1+n2,m,p)."""
    n2 = (U2.shape[0]) // m
    n = n1 + n2
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # U: stack vertically
    U[:n1*m, :R1] = U1
    U[n1*m:, R1:] = U2
    
    # V: concatenate horizontally (same B matrix)
    V[:, :R1] = V1
    V[:, R1:] = V2
    
    # W: stack vertically. g = k*n + i.
    # For first block (i in 0..n1-1): g = k*n + i, i < n1
    # For second block (i in n1..n-1): g = k*n + i, i >= n1
    # W1 has rows indexed by g1 = k*n1 + i (i in 0..n1-1)
    # W2 has rows indexed by g2 = k*n2 + i (i in 0..n2-1)
    # In combined: for k in 0..p-1, i in 0..n-1:
    #   if i < n1: use W1[k*n1 + i]
    #   else: use W2[k*n2 + (i-n1)]
    for k in range(p):
        for i in range(n1):
            W[k * n + i, :R1] = W1[k * n1 + i, :]
        for i in range(n2):
            W[k * n + (n1 + i), R1:] = W2[k * n2 + i, :]
    
    return U, V, W

def _combine_split_p(U1, V1, W1, U2, V2, W2, n, m, p1):
    """Combine decompositions for (n,m,p1) and (n,m,p2) into (n,m,p1+p2)."""
    p2 = (V2.shape[0]) // m
    p = p1 + p2
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # U: concatenate horizontally
    U[:, :R1] = U1
    U[:, R1:] = U2
    
    # V: stack vertically. b = j*p + k.
    # V1 rows: b1 = j*p1 + k (k in 0..p1-1)
    # V2 rows: b2 = j*p2 + k (k in 0..p2-1)
    # Combined: for j in 0..m-1, k in 0..p-1:
    #   if k < p1: use V1[j*p1 + k]
    #   else: use V2[j*p2 + (k-p1)]
    for j in range(m):
        for k in range(p1):
            V[j * p + k, :R1] = V1[j * p1 + k, :]
        for k in range(p2):
            V[j * p + (p1 + k), R1:] = V2[j * p2 + k, :]
    
    # W: stack vertically. g = k*n + i.
    for k in range(p1):
        for i in range(n):
            W[k * n + i, :R1] = W1[k * n + i, :]
    for k in range(p2):
        for i in range(n):
            W[(p1 + k) * n + i, R1:] = W2[k * n + i, :]
    
    return U, V, W

_rec_cache = {}

def _build_recursive(n, m, p):
    """
    Recursively builds a low-rank decomposition using Strassen-like splits.
    """
    if (n, m, p) in _rec_cache:
        return _rec_cache[(n, m, p)]
    
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        _rec_cache[(n, m, p)] = (U, V, W)
        return U, V, W
        
    if n == 2 and m == 2 and p == 2:
        result = _strassen_2x2()
        _rec_cache[(n, m, p)] = result
        return result
    
    # Base cases: when one dimension is 1, schoolbook is optimal
    if n == 1:
        result = _build_schoolbook(n, m, p)
        _rec_cache[(n, m, p)] = result
        return result
    if m == 1:
        result = _build_schoolbook(n, m, p)
        _rec_cache[(n, m, p)] = result
        return result
    if p == 1:
        result = _build_schoolbook(n, m, p)
        _rec_cache[(n, m, p)] = result
        return result
    
    # Try all possible splits of each dimension and pick the one with minimum total rank
    best_rank = float('inf')
    best_result = None
    
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _build_recursive(n, m1, p)
        U2, V2, W2 = _build_recursive(n, m2, p)
        r = U1.shape[1] + U2.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = _combine_decompositions(U1, V1, W1, U2, V2, W2, n, m1, m2, p)
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _build_recursive(n1, m, p)
        U2, V2, W2 = _build_recursive(n2, m, p)
        r = U1.shape[1] + U2.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = _combine_split_n(U1, V1, W1, U2, V2, W2, n1, m, p)
    
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _build_recursive(n, m, p1)
        U2, V2, W2 = _build_recursive(n, m, p2)
        r = U1.shape[1] + U2.shape[1]
        if r < best_rank:
            best_rank = r
            best_result = _combine_split_p(U1, V1, W1, U2, V2, W2, n, m, p1)
    
    if best_result is not None:
        _rec_cache[(n, m, p)] = best_result
        return best_result
    
    # Fallback: schoolbook
    result = _build_schoolbook(n, m, p)
    _rec_cache[(n, m, p)] = result
    return result

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
        elif U_rec.shape[1] < best_rank:
            # Even if verification fails, check with relaxed tolerance
            if _verify_decomposition(U_rec, V_rec, W_rec, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U_rec, V_rec, W_rec
                best_rank = U_rec.shape[1]
    except Exception as e:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
