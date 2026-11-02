import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _build_decomposition(n, m, p):
    """
    Build a decomposition for n x m * m x p using recursive splitting.
    Returns (U, V, W, rank) where:
      U: (n*m, rank) - A indices as i*m+j
      V: (m*p, rank) - B indices as j*p+k  
      W: (p*n, rank) - C indices as k*n+i (trace convention)
    """
    # Base case: if any dimension is 1, use schoolbook for that subproblem
    if n == 1 and m == 1 and p == 1:
        return [[1.0]], [[1.0]], [[1.0]], 1
    
    schoolbook_rank = n * m * p
    
    # If dimensions are small, just use schoolbook
    if n <= 1 or m <= 1 or p <= 1:
        r = n * m * p
        U = [[0.0]*r for _ in range(n*m)]
        V = [[0.0]*r for _ in range(m*p)]
        W = [[0.0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W, r
    
    # Choose which dimension to split: pick the largest one that is >= 2
    # We can split n (output rows), m (middle), or p (output cols)
    
    # Option 1: Split m (middle dimension)
    # A = [A1 | A2], B = [B1; B2], C = A1*B1 + A2*B2
    # Subproblems: n x m1 x p and n x m2 x p
    
    # Option 2: Split n (first matrix rows / output rows)
    # A = [A1; A2], C = [C1; C2], C1 = A1*B, C2 = A2*B
    # Subproblems: n1 x m x p and n2 x m x p
    
    # Option 3: Split p (second matrix cols / output cols)
    # B = [B1 | B2], C = [C1 | C2], C1 = A*B1, C2 = A*B2
    # Subproblems: n x m x p1 and n x m x p2
    
    # Estimate ranks for each option
    def estimate_rank_split_m():
        if m < 2:
            return n * m * p
        m1 = m // 2
        m2 = m - m1
        _, _, _, r1 = _build_decomposition(n, m1, p)
        _, _, _, r2 = _build_decomposition(n, m2, p)
        return r1 + r2
    
    def estimate_rank_split_n():
        if n < 2:
            return n * m * p
        n1 = n // 2
        n2 = n - n1
        _, _, _, r1 = _build_decomposition(n1, m, p)
        _, _, _, r2 = _build_decomposition(n2, m, p)
        return r1 + r2
    
    def estimate_rank_split_p():
        if p < 2:
            return n * m * p
        p1 = p // 2
        p2 = p - p1
        _, _, _, r1 = _build_decomposition(n, m, p1)
        _, _, _, r2 = _build_decomposition(n, m, p2)
        return r1 + r2
    
    # Pick the best split
    candidates = []
    if m >= 2:
        candidates.append(('m', estimate_rank_split_m()))
    if n >= 2:
        candidates.append(('n', estimate_rank_split_n()))
    if p >= 2:
        candidates.append(('p', estimate_rank_split_p()))
    
    if not candidates:
        # All dimensions are 1, but we handled that above
        r = n * m * p
        U = [[0.0]*r for _ in range(n*m)]
        V = [[0.0]*r for _ in range(m*p)]
        W = [[0.0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W, r
    
    # Sort by estimated rank
    candidates.sort(key=lambda x: x[1])
    best_split = candidates[0][0]
    est_rank = candidates[0][1]
    
    # If estimated rank is not better than schoolbook, use schoolbook
    if est_rank >= schoolbook_rank:
        r = n * m * p
        U = [[0.0]*r for _ in range(n*m)]
        V = [[0.0]*r for _ in range(m*p)]
        W = [[0.0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W, r
    
    # Perform the split
    U = [[0.0]*est_rank for _ in range(n*m)]
    V = [[0.0]*est_rank for _ in range(m*p)]
    W = [[0.0]*est_rank for _ in range(p*n)]
    
    if best_split == 'm':
        m1 = m // 2
        m2 = m - m1
        U1, V1, W1, r1 = _build_decomposition(n, m1, p)
        U2, V2, W2, r2 = _build_decomposition(n, m2, p)
        
        # Embed block 1: A1 (n x m1), B1 (m1 x p)
        for i in range(n):
            for j in range(m1):
                for r_idx in range(r1):
                    U[i*m + j][r_idx] += U1[i*m1 + j][r_idx]
        for j in range(m1):
            for k in range(p):
                for r_idx in range(r1):
                    V[j*p + k][r_idx] += V1[j*p + k][r_idx]
        for k in range(p):
            for i in range(n):
                for r_idx in range(r1):
                    W[k*n + i][r_idx] += W1[k*n + i][r_idx]
        
        # Embed block 2: A2 (n x m2), B2 (m2 x p)
        for i in range(n):
            for j in range(m2):
                for r_idx in range(r2):
                    U[i*m + (m1 + j)][r1 + r_idx] += U2[i*m2 + j][r_idx]
        for j in range(m2):
            for k in range(p):
                for r_idx in range(r2):
                    V[(m1 + j)*p + k][r1 + r_idx] += V2[j*p + k][r_idx]
        for k in range(p):
            for i in range(n):
                for r_idx in range(r2):
                    W[k*n + i][r1 + r_idx] += W2[k*n + i][r_idx]
                    
    elif best_split == 'n':
        n1 = n // 2
        n2 = n - n1
        U1, V1, W1, r1 = _build_decomposition(n1, m, p)
        U2, V2, W2, r2 = _build_decomposition(n2, m, p)
        
        # Embed block 1: A1 (n1 x m), B (m x p), C1 (p x n1)
        for i in range(n1):
            for j in range(m):
                for r_idx in range(r1):
                    U[i*m + j][r_idx] += U1[i*m + j][r_idx]
        for j in range(m):
            for k in range(p):
                for r_idx in range(r1):
                    V[j*p + k][r_idx] += V1[j*p + k][r_idx]
        for k in range(p):
            for i in range(n1):
                for r_idx in range(r1):
                    W[k*n + i][r_idx] += W1[k*n1 + i][r_idx]
        
        # Embed block 2: A2 (n2 x m), B (m x p), C2 (p x n2)
        for i in range(n2):
            for j in range(m):
                for r_idx in range(r2):
                    U[(n1 + i)*m + j][r1 + r_idx] += U2[i*m + j][r_idx]
        for j in range(m):
            for k in range(p):
                for r_idx in range(r2):
                    V[j*p + k][r1 + r_idx] += V2[j*p + k][r_idx]
        for k in range(p):
            for i in range(n2):
                for r_idx in range(r2):
                    W[k*n + (n1 + i)][r1 + r_idx] += W2[k*n2 + i][r_idx]
                    
    elif best_split == 'p':
        p1 = p // 2
        p2 = p - p1
        U1, V1, W1, r1 = _build_decomposition(n, m, p1)
        U2, V2, W2, r2 = _build_decomposition(n, m, p2)
        
        # Embed block 1: A (n x m), B1 (m x p1), C1 (p1 x n)
        for i in range(n):
            for j in range(m):
                for r_idx in range(r1):
                    U[i*m + j][r_idx] += U1[i*m + j][r_idx]
        for j in range(m):
            for k in range(p1):
                for r_idx in range(r1):
                    V[j*p + k][r_idx] += V1[j*p1 + k][r_idx]
        for k in range(p1):
            for i in range(n):
                for r_idx in range(r1):
                    W[k*n + i][r_idx] += W1[k*n + i][r_idx]
        
        # Embed block 2: A (n x m), B2 (m x p2), C2 (p2 x n)
        for i in range(n):
            for j in range(m):
                for r_idx in range(r2):
                    U[i*m + j][r1 + r_idx] += U2[i*m + j][r_idx]
        for j in range(m):
            for k in range(p2):
                for r_idx in range(r2):
                    V[j*p + (p1 + k)][r1 + r_idx] += V2[j*p2 + k][r_idx]
        for k in range(p2):
            for i in range(n):
                for r_idx in range(r2):
                    W[(p1 + k)*n + i][r1 + r_idx] += W2[k*n + i][r_idx]
    
    return U, V, W, est_rank


def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r_idx in range(r):
                    total += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                if abs(total - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the recursive decomposition
    U, V, W, rank = _build_decomposition(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
