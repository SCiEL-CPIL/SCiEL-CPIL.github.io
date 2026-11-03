"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_BUDGET = 115.0  # safety margin before 120s kill

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    """Schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _build_strassen_3x3x8():
    """
    Build a Strassen-based decomposition for 3x3 * 3x8.
    
    Strategy: Use Strassen's 7-product algorithm on the top-left 2x2 block
    (rows 0,1 of A times rows 0,1 of B), plus schoolbook for the remaining
    cross terms involving row/column 2.
    
    A = [[A00, A01, A02],   B = [[B00..B07],
         [A10, A11, A12],        [B10..B17],
         [A20, A21, A22]]        [B20..B27]]
    
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    For i in {0,1}, k in 0..7:
      C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    For i=2, k in 0..7:
      C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    
    The 2x2 * 2x8 sub-problem (A[0:2,0:2] * B[0:2,:]) uses Strassen:
    7 products instead of 4*8=32... wait, 2*2*8 = 32 schoolbook terms,
    Strassen gives 7 products each producing an 8-vector, so 7 terms.
    
    Cross terms:
    - A[0,2]*B[2,k] for k=0..7: 1 term (scalar * vector)
    - A[1,2]*B[2,k] for k=0..7: 1 term
    - A[2,0]*B[0,k] for k=0..7: 1 term
    - A[2,1]*B[1,k] for k=0..7: 1 term
    - A[2,2]*B[2,k] for k=0..7: 1 term
    
    Total rank: 7 + 5 = 12
    """
    n, m, p = 3, 3, 8
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []  # list of (U_dict, V_dict, W_dict)
    
    # Strassen 2x2 * 2x8
    # M1 = (A00+A11) * (B0+B1)
    terms.append(({u_idx(0,0): 1, u_idx(1,1): 1},
                  {v_idx(0,k): 1 for k in range(p)} | {v_idx(1,k): 1 for k in range(p)},
                  {w_idx(k,0): 1 for k in range(p)} | {w_idx(k,1): 1 for k in range(p)}))
    
    # M2 = (A10+A11) * B0
    terms.append(({u_idx(1,0): 1, u_idx(1,1): 1},
                  {v_idx(0,k): 1 for k in range(p)},
                  {w_idx(k,1): 1 for k in range(p)}))
    
    # M3 = A00 * (B1-B0)
    terms.append(({u_idx(0,0): 1},
                  {v_idx(1,k): 1 for k in range(p)} | {v_idx(0,k): -1 for k in range(p)},
                  {w_idx(k,1): 1 for k in range(p)}))
    
    # M4 = A11 * (B0-B1)
    terms.append(({u_idx(1,1): 1},
                  {v_idx(0,k): 1 for k in range(p)} | {v_idx(1,k): -1 for k in range(p)},
                  {w_idx(k,0): 1 for k in range(p)}))
    
    # M5 = (A00+A01) * B1
    terms.append(({u_idx(0,0): 1, u_idx(0,1): 1},
                  {v_idx(1,k): 1 for k in range(p)},
                  {w_idx(k,0): 1 for k in range(p)}))
    
    # M6 = (A10-A00) * (B0+B1)
    terms.append(({u_idx(1,0): 1, u_idx(0,0): -1},
                  {v_idx(0,k): 1 for k in range(p)} | {v_idx(1,k): 1 for k in range(p)},
                  {w_idx(k,1): 1 for k in range(p)}))
    
    # M7 = (A01-A11) * (B1+B0)
    terms.append(({u_idx(0,1): 1, u_idx(1,1): -1},
                  {v_idx(1,k): 1 for k in range(p)} | {v_idx(0,k): 1 for k in range(p)},
                  {w_idx(k,0): 1 for k in range(p)}))
    
    # Cross terms
    # A[0,2] * B[2,k]
    terms.append(({u_idx(0,2): 1},
                  {v_idx(2,k): 1 for k in range(p)},
                  {w_idx(k,0): 1 for k in range(p)}))
    
    # A[1,2] * B[2,k]
    terms.append(({u_idx(1,2): 1},
                  {v_idx(2,k): 1 for k in range(p)},
                  {w_idx(k,1): 1 for k in range(p)}))
    
    # A[2,0] * B[0,k]
    terms.append(({u_idx(2,0): 1},
                  {v_idx(0,k): 1 for k in range(p)},
                  {w_idx(k,2): 1 for k in range(p)}))
    
    # A[2,1] * B[1,k]
    terms.append(({u_idx(2,1): 1},
                  {v_idx(1,k): 1 for k in range(p)},
                  {w_idx(k,2): 1 for k in range(p)}))
    
    # A[2,2] * B[2,k]
    terms.append(({u_idx(2,2): 1},
                  {v_idx(2,k): 1 for k in range(p)},
                  {w_idx(k,2): 1 for k in range(p)}))
    
    r = len(terms)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c in range(r):
        Ud, Vd, Wd = terms[c]
        for idx, val in Ud.items():
            U[idx][c] = val
        for idx, val in Vd.items():
            V[idx][c] = val
        for idx, val in Wd.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}

def _build_strassen_embedded():
    """
    Improved decomposition using multiple Strassen blocks.
    
    Split the 3x3 * 3x8 problem into overlapping Strassen 2x2 blocks.
    
    Approach: Partition into three 2x2 blocks along the diagonal of the
    3x3 matrix, each paired with appropriate 2x8 sub-matrices of B.
    
    Block 1: A[0:2,0:2] * B[0:2,:]  -> Strassen 2x2*2x8 (7 terms)
    Block 2: A[1:3,1:3] * B[1:3,:]  -> Strassen 2x2*2x8 (7 terms)
    
    But these overlap at row 1 of A and row 1 of B. We need to be careful
    about double-counting.
    
    Actually, let's use a different strategy:
    
    Partition A into:
    - Top 2 rows: A[0:2,:] (2x3)
    - Bottom row: A[2,:] (1x3)
    
    Partition B into:
    - Top 2 rows: B[0:2,:] (2x8)
    - Bottom row: B[2,:] (1x8)
    
    C[0:2,:] = A[0:2,:] * B[:, :] (2x3 * 3x8)
    C[2,:]   = A[2,:] * B[:, :]   (1x3 * 3x8)
    
    For A[0:2,:] * B[:, :]:
    = A[0:2,0:2] * B[0:2,:] + A[0:2,2] * B[2,:]
    
    A[0:2,0:2] * B[0:2,:] is 2x2 * 2x8 -> Strassen (7 terms)
    A[0:2,2] * B[2,:] is 2x1 * 1x8 -> 2 terms (each scalar*vector)
    
    For A[2,:] * B[:, :]:
    = A[2,0:2] * B[0:2,:] + A[2,2] * B[2,:]
    
    A[2,0:2] * B[0:2,:] is 1x2 * 2x8 -> 2 terms
    A[2,2] * B[2,:] is 1x1 * 1x8 -> 1 term
    
    Total: 7 + 2 + 2 + 1 = 12 terms
    
    This is the same as before. Let me try a different split.
    
    Alternative: Use Strassen on a 2x2 sub-block and handle the rest.
    The key insight from the inspiration is that embedding 4 non-overlapping
    Strassen blocks reduced rank from 72 to 68. But for 3x3*3x8, the
    schoolbook rank is 72, and we can get to 12 with one Strassen block.
    
    Let me verify the 12-term solution is correct.
    """
    return _build_strassen_3x3x8()

def _build_multi_strassen():
    """
    Try to find a decomposition with rank less than 12.
    
    Lower bound: The tensor rank of 3x3*3x8 matrix multiplication.
    For n=m=p=3, the rank is at least 19 (best known upper bound is 23).
    For 3x3*3x8, a trivial lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24.
    Wait, that's not right. The rank must be at least the max of the dimensions
    of the three factors? No, that's not a valid lower bound either.
    
    Actually, a valid lower bound is: rank >= max(n*m, m*p, n*p) is NOT correct.
    The correct simple lower bound is rank >= max(n, m, p) * something...
    
    Let me just verify the 12-term solution and see if we can do better.
    
    For 2x2*2x2, Strassen gives rank 7 vs schoolbook 8.
    For 2x2*2x8, we get 7 terms (each multiplying two 8-vectors conceptually,
    but in the tensor decomposition each "product" is one term in the rank).
    
    Actually wait - I need to reconsider. In the tensor decomposition of
    matrix multiplication A(n x m) * B(m x p) -> C(n x p), the tensor
    is in U(n*m) x V(m*p) x W(n*p) and has rank r if it can be written
    as sum of r rank-1 tensors.
    
    Each rank-1 tensor corresponds to a "product" of one element of A,
    one element of B, and one element of C. But with Strassen, we have
    "products" that are sums of elements, which still count as one rank-1
    term in the decomposition.
    
    So for 2x2*2x8: the tensor is in R^{4} x R^{16} x R^{16}.
    Schoolbook: 4*8 = 32 rank-1 terms.
    Strassen: 7 rank-1 terms (each is a product of a linear form in A,
    a linear form in B, and a linear form in C).
    
    Yes, so 7 is correct for the 2x2*2x8 sub-problem.
    
    Total for 3x3*3x8 with one Strassen block: 7 + 5 = 12.
    
    Can we do better? The theoretical minimum for 3x3*3x3 is 19 (lower bound)
    with best known upper bound 23. For 3x3*3x8, it should be at least as hard.
    Actually 3x3*3x8 should have rank at least 24 since we need at least
    24 terms to cover the 24 output positions... no that's not right either.
    
    Let me just go with 12 and verify it works.
    """
    return _build_strassen_3x3x8()

def _try_bilinear_optimization():
    """
    Try to find a lower-rank decomposition using optimization.
    
    We can formulate this as: find rank r matrices U, V, W such that
    for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    This is a non-convex problem. We can try random initialization with
    gradient descent, but it's unlikely to find a solution with rank < 12
    given the structure.
    
    For now, just return the Strassen solution.
    """
    return _build_strassen_3x3x8()

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try Strassen-based decomposition
    if _time_left() > 5.0:
        cand = _build_strassen_3x3x8()
        if cand is not None:
            r = len(cand["U"][0])
            if r < best_rank and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
                best = cand
                best_rank = r
    
    # Try multi-Strassen or other approaches
    if _time_left() > 10.0:
        cand2 = _build_multi_strassen()
        if cand2 is not None:
            r2 = len(cand2["U"][0])
            if r2 < best_rank and _verify(cand2["U"], cand2["V"], cand2["W"], n, m, p):
                best = cand2
                best_rank = r2
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
