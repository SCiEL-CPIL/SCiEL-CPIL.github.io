import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U: A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V: B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W: C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2: (A10+A11)B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    
    # M3: A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4: A11(B10-B00)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5: (A00+A01)B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _build_decomposed_2x3x3():
    """
    Decompose 2x3x3 into blocks using Strassen where possible.
    Total rank: 17
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x2 block (j in {0,1}, k in {0,1}) -> Strassen rank 7
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # Part 2: 2x2x1 block (j in {0,1}, k=2) -> schoolbook rank 4
    col_offset = 7
    for i in range(2):
        for j in range(2):
            col = col_offset + i * 2 + j
            a_global = i * 3 + j
            b_global = j * 3 + 2
            c_global = 4 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    # Part 3: 2x1x3 block (j=2, k in {0,1,2}) -> schoolbook rank 6
    col_offset = 11
    for i in range(2):
        for k in range(3):
            col = col_offset + i * 3 + k
            a_global = i * 3 + 2
            b_global = 6 + k
            c_global = k * 2 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification of the decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def _try_rank_16_construction():
    """
    Attempt a rank-16 construction using a different block strategy.
    
    Strategy: Use two overlapping 2x2x2 Strassen blocks and correct for overlap.
    Block 1: j in {0,1}, k in {0,1} -> covers (i,j,k) for j in {0,1}, k in {0,1}
    Block 2: j in {1,2}, k in {0,1} -> covers (i,j,k) for j in {1,2}, k in {0,1}
    Overlap: j=1, k in {0,1}
    
    Then handle k=2 separately.
    """
    n, m, p = 2, 3, 3
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: j in {0,1}, k in {0,1}
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # Block 2: j in {1,2}, k in {0,1}
    # Map: j_local 0->1, 1->2; k_local 0->0, 1->1
    for r in range(7):
        col = 7 + r
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            j_global = 1 + j_local  # j_local 0->1, 1->2
            a_global = i_local * 3 + j_global
            U[a_global][col] = U1[a_idx][r]
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            j_global = 1 + j_local
            b_global = j_global * 3 + k_local
            V[b_global][col] = V1[b_idx][r]
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][col] = W1[c_idx][r]
    
    # Now we have double-counted j=1, k in {0,1}. Need to subtract.
    # The overlap terms are: for each (i, k in {0,1}), the contribution from j=1 is doubled.
    # We need to subtract one copy of A_{i,1} * B_{1,k} * C_{k,i} for each i, k.
    # That's 2*2 = 4 terms to subtract.
    # But we only have 16-14 = 2 slots left. This doesn't work.
    
    return None  # This approach doesn't fit in rank 16

def _try_direct_rank_15():
    """
    Try a direct rank-15 construction. This is based on the idea that
    2x3x3 might have a lower rank than 17.
    
    We'll set up the tensor equation and try to solve it.
    """
    # This is a placeholder for a more sophisticated approach.
    # For now, return None.
    return None

def _random_search_rank_16(max_time=30.0):
    """
    Random search for a rank-16 decomposition.
    """
    n, m, p = 2, 3, 3
    rank = 16
    start_time = time.time()
    
    for attempt in range(10000):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with random small values
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Start from the rank-17 solution and try to merge two columns
        U17, V17, W17 = _build_decomposed_2x3x3()
        U = np.array(U17, dtype=float)[:, :15]  # Take first 15 columns
        V = np.array(V17, dtype=float)[:, :15]
        W = np.array(W17, dtype=float)[:, :15]
        
        # Try to add a 16th column that corrects the missing terms
        # The missing terms are from columns 15 and 16 of the rank-17 solution
        # We need to find if their sum can be represented as a single rank-1 term
        # or if we can adjust the existing columns.
        
        # This is a heuristic approach; it's unlikely to succeed.
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # First: try the known rank-17 decomposition
    try:
        U, V, W = _build_decomposed_2x3x3()
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
