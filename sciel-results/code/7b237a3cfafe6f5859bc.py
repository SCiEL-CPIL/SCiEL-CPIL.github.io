"""Tensor decomposition search for 4x4 * 4x8 matrix multiplication.

Uses a structural insight: the 4x4 * 4x8 multiplication can be viewed as
two independent 4x4 * 4x4 multiplications (splitting the 8 output columns
into two groups of 4). If we have a rank-r decomposition for 4x4 * 4x4,
we can get rank 2r for 4x4 * 4x8.

We also attempt to find better-than-schoolbook decompositions via:
1. Known low-rank constructions (Strassen-style for small matrices)
2. Randomized algebraic search with verification
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130  # 20s safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the matmul tensor.
    
    Constraint: for all i, k, j, j':
        sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta(j, j')
    """
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    if j == jp:
                        expected = 1.0
                    else:
                        expected = 0.0
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    if abs(val - expected) > tol:
                        return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # For each i, k: the m x m matrix M_{i,k} = sum_c W[k*n+i,c] * U_i[:,c] * V_k[:,c]^T
    # should equal identity, where U_i is the m-vector U[i*m:(i+1)*m, :]
    # and V_k is the m-vector V[:, k] (columns of V corresponding to output k)
    # Actually V has shape (m*p, rank), so V[j*p+k, c] for fixed k is V[k::p, c]
    for i in range(n):
        U_i = U[i * m:(i + 1) * m, :]  # shape (m, rank)
        for k in range(p):
            V_k = V[k::p, :]  # shape (m, rank), entries V[j*p+k, c]
            # M = sum_c W[k*n+i, c] * U_i[:, c] * V_k[:, c]^T
            #   = U_i * diag(W_kn+i) * V_k^T
            w = W[k * n + i, :]  # shape (rank,)
            M = U_i * w[None, :] @ V_k.T  # shape (m, m)
            if np.max(np.abs(M - np.eye(m))) > tol:
                return False
    return True


def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
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
    """Return Strassen rank-7 decomposition for 2x2 * 2x2.
    
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+g)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # U[i*2+j, c] = coefficient of A[i,j] in M_c
    # V[j*2+k, c] = coefficient of B[j,k] in M_c
    # W[k*2+i, c] = coefficient of M_c in C[i,k]
    
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C[0,0]+=1, C[1,1]+=1
    U[0, 0] = 1; U[3, 0] = 1  # a, d
    V[0, 0] = 1; V[3, 0] = 1  # e, h
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: C[1,0]+=1, C[1,1]-=1
    U[2, 1] = 1; U[3, 1] = 1  # c, d
    V[0, 1] = 1  # e
    W[2, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]
    
    # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: C[0,1]+=1, C[1,1]+=1
    U[0, 2] = 1  # a
    V[1, 2] = 1; V[3, 2] = -1  # f, h
    W[1, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # M4 = d(g-e): U: d=1; V: g=1,e=-1; W: C[0,0]+=1, C[1,0]+=1
    U[3, 3] = 1  # d
    V[2, 3] = 1; V[0, 3] = -1  # g, e
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0], C[1,0]
    
    # M5 = (a+b)h: U: a=1,b=1; V: h=1; W: C[0,0]-=1, C[0,1]+=1
    U[0, 4] = 1; U[1, 4] = 1  # a, b
    V[3, 4] = 1  # h
    W[0, 4] = -1; W[1, 4] = 1  # C[0,0], C[0,1]
    
    # M6 = (c-d)(f+g): U: c=1,d=-1; V: f=1,g=1; W: C[1,1]+=1
    U[2, 5] = 1; U[3, 5] = -1  # c, d
    V[1, 5] = 1; V[2, 5] = 1  # f, g
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (b-d)(e+g): U: b=1,d=-1; V: e=1,g=1; W: C[0,0]+=1
    U[1, 6] = 1; U[3, 6] = -1  # b, d
    V[0, 6] = 1; V[2, 6] = 1  # e, g
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _combine_two_2x4(U2, V2, W2, n=4, m=4, p=4):
    """Build 4x4 * 4x4 decomposition using 4 independent 2x2 block products.
    
    A 4x4 * 4x4 multiplication decomposes into 4 block multiplications:
    C[ci,cj] = sum_bk A[ci,bk] * B[bk,cj] for ci,cj in {0,1}
    
    For each (ci, bk, cj) triple, we compute A[ci,bk] * B[bk,cj] using Strassen rank-7.
    Total: 4 (for ci,cj) * 2 (for bk) * 7 = 56 rank terms.
    But we can share: for fixed (ci, bk), the result contributes to C[ci,0] and C[ci,1].
    Actually each (ci, bk, cj) is independent, so 2*2*2*7 = 56.
    Wait: ci in {0,1}, bk in {0,1}, cj in {0,1} -> 8 combinations, each rank 7 -> rank 56.
    
    But we can do better: for each (ci, cj), C[ci,cj] = A[ci,0]*B[0,cj] + A[ci,1]*B[1,cj].
    This is a sum of two 2x2 products. Each 2x2 product has rank 7.
    So for each (ci,cj), we need 2*7 = 14 rank. Total: 4 * 14 = 56.
    
    Actually the minimum is: 4 output blocks, each needs sum of 2 products of rank 7 each.
    Since the products are independent, total rank = 4 * 2 * 7 = 56.
    """
    rank = 4 * 2 * 7  # 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for ci in range(2):  # output row block
        for cj in range(2):  # output col block
            for bk in range(2):  # middle block
                # A[ci,bk] * B[bk,cj] contributes to C[ci,cj]
                # Use Strassen 2x2 for this block product
                for s in range(7):
                    # U for A block (ci, bk): rows 2*ci to 2*ci+1, cols 2*bk to 2*bk+1
                    for di in range(2):
                        for dj in range(2):
                            U[(2*ci+di)*m + (2*bk+dj), col] = U2[di*2+dj, s]
                    
                    # V for B block (bk, cj): rows 2*bk to 2*bk+1, cols 2*cj to 2*cj+1
                    for di in range(2):
                        for dj in range(2):
                            V[(2*bk+di)*p + (2*cj+dj), col] = V2[di*2+dj, s]
                    
                    # W for C block (ci, cj): W[k*n+i, c] where k is col index, i is row index
                    # C row: 2*ci to 2*ci+1, C col: 2*cj to 2*cj+1
                    for di in range(2):
                        for dj in range(2):
                            # di is row offset within block, dj is col offset within block
                            W[(2*cj+dj)*n + (2*ci+di), col] = W2[di*2+dj, s]
                    
                    col += 1
    
    return U, V, W


def _two_2x4_blocks(U2, V2, W2):
    """Construct 4x4 * 4x8 as two independent 4x4 * 4x4 multiplications.
    
    Split the 8 output columns into two groups of 4.
    Each group is a 4x4 * 4x4 multiplication.
    Use the 4x4*4x4 decomposition for each.
    """
    n, m, p = 4, 4, 8
    
    # Get 4x4 * 4x4 decomposition (rank 56)
    U4, V4, W4 = _combine_two_2x4(U2, V2, W2)
    rank4 = U4.shape[1]
    
    # For 4x4 * 4x8:
    # C[:, 0:4] = A * B[:, 0:4]  (first 4x4)
    # C[:, 4:8] = A * B[:, 4:8]  (second 4x4)
    
    rank = 2 * rank4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First 4x4 block (columns 0-3 of output)
    U[:, 0:rank4] = U4
    # V for first block: B[:, 0:4], so V[j*8+k, c] for k in 0..3
    for j in range(m):
        for k in range(4):
            V[j * p + k, 0:rank4] = V4[j * 4 + k, :]
    # W for first block: C[:, 0:4], so W[k*n+i, c] for k in 0..3
    for k in range(4):
        for i in range(n):
            W[k * n + i, 0:rank4] = W4[k * 4 + i, :]
    
    # Second 4x4 block (columns 4-7 of output)
    U[:, rank4:2*rank4] = U4
    # V for second block: B[:, 4:8], so V[j*8+k, c] for k in 4..7
    for j in range(m):
        for k in range(4):
            V[j * p + (k + 4), rank4:2*rank4] = V4[j * 4 + k, :]
    # W for second block: C[:, 4:8], so W[k*n+i, c] for k in 4..7
    for k in range(4):
        for i in range(n):
            W[(k + 4) * n + i, rank4:2*rank4] = W4[k * 4 + i, :]
    
    return U, V, W


def _random_search(n, m, p, target_rank, n_trials=50, time_budget=60):
    """Try random rank-r decompositions and verify."""
    start = time.time()
    rng = np.random.RandomState(42)
    
    for trial in range(n_trials):
        if time.time() - start > time_budget:
            break
        
        U = rng.randn(n * m, target_rank)
        V = rng.randn(m * p, target_rank)
        W = rng.randn(p * n, target_rank)
        
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None


def _greedy_rank_reduction(n, m, p, time_budget=30):
    """Try to reduce rank from schoolbook by removing columns.
    
    Start with schoolbook, try removing each column and check if the
    system can still be solved (i.e., the remaining columns still span).
    """
    start = time.time()
    U, V, W = _schoolbook(n, m, p)
    rank = U.shape[1]
    
    # For each column, try removing it and checking if we can fix the rest
    # This is complex; skip for now
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        # Emergency fallback
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = None
    best_rank = n * m * p  # 128
    
    # Strategy 1: Use Strassen-based construction
    if time.time() < _DEADLINE:
        U2, V2, W2 = _strassen_2x2()
        # Verify Strassen 2x2
        if _verify_fast(U2, V2, W2, 2, 2, 2):
            # Build 4x4 * 4x8 from two 4x4 * 4x4 blocks
            U, V, W = _two_2x4_blocks(U2, V2, W2)
            if _verify_fast(U, V, W, n, m, p):
                best = (U, V, W)
                best_rank = U.shape[1]
                print(f"Strassen-based: rank {best_rank}")
            else:
                print("Strassen-based construction failed verification")
        else:
            print("Strassen 2x2 failed verification")
    
    # Strategy 2: Schoolbook as fallback
    if best is None:
        U, V, W = _schoolbook(n, m, p)
        best = (U, V, W)
        best_rank = n * m * p
    
    # Strategy 3: Try to find even better with more time
    if time.time() < _DEADLINE - 20 and best_rank > 60:
        # Try random search for lower rank
        for target_rank in range(min(best_rank - 1, 58), 40, -2):
            if time.time() > _DEADLINE - 10:
                break
            result = _random_search(n, m, p, target_rank, n_trials=5, 
                                   time_budget=min(8, _DEADLINE - time.time() - 5))
            if result is not None:
                U, V, W = result
                best = (U, V, W)
                best_rank = target_rank
                print(f"Random search found rank {target_rank}")
                break
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
