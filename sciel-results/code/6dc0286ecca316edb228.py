"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    # First, get the guaranteed-valid schoolbook decomposition
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    
    # Now try to improve with Strassen-based block decomposition
    # Strategy: Split into blocks and use Strassen where 2x2 blocks exist
    # For 3x5x6:
    #   - i: {0,1} and {2}  (top 2 rows, bottom 1 row)
    #   - j: {0,1}, {2,3}, {4}  
    #   - k: {0,1}, {2,3}, {4,5}
    # 
    # For the top 2 rows (i in {0,1}), we can use Strassen for 2x2x2 blocks
    # For the bottom row (i=2), we must use schoolbook (1x2x2 blocks)
    
    best_result = None
    best_rank = rank_school
    
    # Try Strassen-based decomposition
    if time.time() < deadline:
        result = _try_strassen_3x5x6(n, m, p)
        if result is not None:
            U, V, W, rank = result
            if _verify_fast(U, V, W, n, m, p):
                best_result = (U, V, W)
                best_rank = rank
    
    # Try alternative block splits
    if time.time() < deadline:
        # Split j as {0}, {1,2}, {3,4} and k as {0,1}, {2,3}, {4,5}
        # For top 2 rows: Strassen on 2x2 (j-blocks {1,2} with k-blocks)
        result = _try_strassen_alt_split(n, m, p)
        if result is not None:
            U, V, W, rank = result
            if _verify_fast(U, V, W, n, m, p) and rank < best_rank:
                best_result = (U, V, W)
                best_rank = rank
    
    # Use the best result found
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_school, V_school, W_school
    
    # Convert to list of lists with integer/half-integer entries
    U_list = _to_list(U)
    V_list = _to_list(V)
    W_list = _to_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _to_list(matrix):
    """Convert numpy matrix to list of lists, rounding to integer/half-integer."""
    result = []
    for row in matrix:
        new_row = []
        for x in row:
            rx = round(x * 2) / 2  # round to nearest half-integer
            if abs(x - rx) < 1e-9:
                if rx == int(rx):
                    new_row.append(int(rx))
                else:
                    new_row.append(rx)
            else:
                new_row.append(float(x))
        result.append(new_row)
    return result


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition: rank = n*m*p."""
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
    
    return U, V, W, rank


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: 4x7 (rows = i*m+j for i,j in {0,1}, m=2)
    V: 4x7 (rows = j*p+k for j,k in {0,1}, p=2)  
    W: 4x7 (rows = k*n+i for k,i in {0,1}, n=2)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications:
    # m0 = (a+d)(e+h)
    # m1 = (c+d)e
    # m2 = a(f-h)
    # m3 = d(g-e)
    # m4 = (a+b)(h)
    # m5 = (c-d)(g+h)
    # m6 = (a-c)(e+f)
    #
    # C[0,0] = m0 + m3 - m4 - m6
    # C[0,1] = m2 + m4
    # C[1,0] = m1 + m3
    # C[1,1] = m0 - m1 + m2 + m5
    
    # U factors (A matrix contributions)
    # Row indices: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6
    
    # V factors (B matrix contributions)
    # Row indices: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6
    
    # W factors (C output contributions)
    # Row indices: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
    # C[0,0] = m0 + m3 - m4 - m6
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = -1  # -m6 in C[0,0]
    # C[0,1] = m2 + m4
    W[2, 2] = 1   # m2 in C[0,1]
    W[2, 4] = 1   # m4 in C[0,1]
    # C[1,0] = m1 + m3
    W[1, 1] = 1   # m1 in C[1,0]
    W[1, 3] = 1   # m3 in C[1,0]
    # C[1,1] = m0 - m1 + m2 + m5
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _try_strassen_3x5x6(n, m, p):
    """
    Try Strassen-based decomposition for 3x5x6.
    
    Block structure:
    - i: top={0,1}, bottom={2}
    - j: blocks {0,1}, {2,3}, {4}
    - k: blocks {0,1}, {2,3}, {4,5}
    
    For top 2 rows (i in {0,1}):
      - j-block {0,1}: 2x2x2 per k-block -> Strassen (rank 7)
      - j-block {2,3}: 2x2x2 per k-block -> Strassen (rank 7)
      - j-block {4}:   2x1x2 per k-block -> schoolbook (rank 2)
    
    For bottom row (i=2):
      - j-block {0,1}: 1x2x2 per k-block -> schoolbook (rank 2)
      - j-block {2,3}: 1x2x2 per k-block -> schoolbook (rank 2)
      - j-block {4}:   1x1x2 per k-block -> schoolbook (rank 2)
    
    Per k-block: (7+7+2) + (2+2+2) = 16 + 6 = 22
    Total: 3 * 22 = 66
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Per k-block:
    #   Top: 7 (j{0,1}) + 7 (j{2,3}) + 2 (j{4}) = 16
    #   Bottom: 2 (j{0,1}) + 2 (j{2,3}) + 2 (j{4}) = 6
    #   Total per k-block: 22
    # 3 k-blocks: 66
    rank = 66
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            U[0 * 5 + 0, col] = sU[0, c]   # i=0, j=0
            U[0 * 5 + 1, col] = sU[1, c]   # i=0, j=1
            U[1 * 5 + 0, col] = sU[2, c]   # i=1, j=0
            U[1 * 5 + 1, col] = sU[3, c]   # i=1, j=1
            V[0 * 6 + k0, col] = sV[0, c]  # j=0, k=k0
            V[0 * 6 + k1, col] = sV[1, c]  # j=0, k=k1
            V[1 * 6 + k0, col] = sV[2, c]  # j=1, k=k0
            V[1 * 6 + k1, col] = sV[3, c]  # j=1, k=k1
            W[k0 * 3 + 0, col] = sW[0, c]  # k=k0, i=0
            W[k0 * 3 + 1, col] = sW[1, c]  # k=k0, i=1
            W[k1 * 3 + 0, col] = sW[2, c]  # k=k1, i=0
            W[k1 * 3 + 1, col] = sW[3, c]  # k=k1, i=1
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0 * 5 + 2, col] = sU[0, c]
            U[0 * 5 + 3, col] = sU[1, c]
            U[1 * 5 + 2, col] = sU[2, c]
            U[1 * 5 + 3, col] = sU[3, c]
            V[2 * 6 + k0, col] = sV[0, c]
            V[2 * 6 + k1, col] = sV[1, c]
            V[3 * 6 + k0, col] = sV[2, c]
            V[3 * 6 + k1, col] = sV[3, c]
            W[k0 * 3 + 0, col] = sW[0, c]
            W[k0 * 3 + 1, col] = sW[1, c]
            W[k1 * 3 + 0, col] = sW[2, c]
            W[k1 * 3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        # Two rank-1 terms: one for k=k0, one for k=k1
        for k_idx in [k0, k1]:
            U[0 * 5 + 4, col] = 1
            U[1 * 5 + 4, col] = 1
            V[4 * 6 + k_idx, col] = 1
            W[k_idx * 3 + 0, col] = 1
            W[k_idx * 3 + 1, col] = 1
            col += 1
        
        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for k_idx in [k0, k1]:
            U[2 * 5 + 0, col] = 1
            U[2 * 5 + 1, col] = 1
            V[0 * 6 + k_idx, col] = 1
            V[1 * 6 + k_idx, col] = 1
            W[k_idx * 3 + 2, col] = 1
            col += 1
        
        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for k_idx in [k0, k1]:
            U[2 * 5 + 2, col] = 1
            U[2 * 5 + 3, col] = 1
            V[2 * 6 + k_idx, col] = 1
            V[3 * 6 + k_idx, col] = 1
            W[k_idx * 3 + 2, col] = 1
            col += 1
        
        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k_idx in [k0, k1]:
            U[2 * 5 + 4, col] = 1
            V[4 * 6 + k_idx, col] = 1
            W[k_idx * 3 + 2, col] = 1
            col += 1
    
    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _try_strassen_alt_split(n, m, p):
    """
    Alternative block split:
    - i: top={0,1}, bottom={2}
    - j: {0}, {1,2}, {3,4}
    - k: {0,1}, {2,3}, {4,5}
    
    For top 2 rows:
      - j={0}: 2x1x2 per k-block -> schoolbook (rank 2)
      - j={1,2}: 2x2x2 per k-block -> Strassen (rank 7)
      - j={3,4}: 2x2x2 per k-block -> Strassen (rank 7)
    
    For bottom row:
      - j={0}: 1x1x2 per k-block -> schoolbook (rank 2)
      - j={1,2}: 1x2x2 per k-block -> schoolbook (rank 2)
      - j={3,4}: 1x2x2 per k-block -> schoolbook (rank 2)
    
    Per k-block: (2+7+7) + (2+2+2) = 16 + 6 = 22
    Total: 3 * 22 = 66
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    rank = 66
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1
        
        # Top 2 rows, j={0}: schoolbook 2x1x2
        for k_idx in [k0, k1]:
            U[0 * 5 + 0, col] = 1
            U[1 * 5 + 0, col] = 1
            V[0 * 6 + k_idx, col] = 1
            W[k_idx * 3 + 0, col] = 1
            W[k_idx * 3 + 1, col] = 1
            col += 1
        
        # Top 2 rows, j={1,2}: Strassen
        for c in range(7):
            U[0 * 5 + 1, col] = sU[0, c]
            U[0 * 5 + 2, col] = sU[1, c]
            U[1 * 5 + 1, col] = sU[2, c]
            U[1 * 5 + 2, col] = sU[3, c]
            V[1 * 6 + k0, col] = sV[0, c]
            V[1 * 6 + k1, col] = sV[1, c]
            V[2 * 6 + k0, col] = sV[2, c]
            V[2 * 6 + k1, col] = sV[3, c]
            W[k0 * 3 + 0, col] = sW[0, c]
            W[k0 * 3 + 1, col] = sW[1, c]
            W[k1 * 3 + 0, col] = sW[2, c]
            W[k1 * 3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows, j={3,4}: Strassen
        for c in range(7):
            U[0 * 5 + 3, col] = sU[0, c]
            U[0 * 5 + 4, col] = sU[1, c]
            U[1 * 5 + 3, col] = sU[2, c]
            U[1 * 5 + 4, col] = sU[3, c]
            V[3 * 6 + k0, col] = sV[0, c]
            V[3 * 6 + k1, col] = sV[1, c]
            V[4 * 6 + k0, col] = sV[2, c]
            V[4 * 6 + k1, col] = sV[3, c]
            W[k0 * 3 + 0, col] = sW[0, c]
            W[k0 * 3 + 1, col] = sW[1, c]
            W[k1 * 3 + 0, col] = sW[2, c]
            W[k1 * 3 + 1, col] = sW[3, c]
            col += 1
        
        # Bottom row, j={0}: schoolbook 1x1x2
        for k_idx in [k0, k1]:
            U[2 * 5 + 0, col] = 1
            V[0 * 6 + k_idx, col] = 1
            W[k_idx * 3 + 2, col] = 1
            col += 1
        
        # Bottom row, j={1,2}: schoolbook 1x2x2
        for k_idx in [k0, k1]:
            U[2 * 5 + 1, col] = 1
            U[2 * 5 + 2, col] = 1
            V[1 * 6 + k_idx, col] = 1
            V[2 * 6 + k_idx, col] = 1
            W[k_idx * 3 + 2, col] = 1
            col += 1
        
        # Bottom row, j={3,4}: schoolbook 1x2x2
        for k_idx in [k0, k1]:
            U[2 * 5 + 3, col] = 1
            U[2 * 5 + 4, col] = 1
            V[3 * 6 + k_idx, col] = 1
            V[4 * 6 + k_idx, col] = 1
            W[k_idx * 3 + 2, col] = 1
            col += 1
    
    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
