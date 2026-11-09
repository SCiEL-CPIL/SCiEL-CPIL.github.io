"""Program for matrix multiplication 2x4 x 4x7 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _target_tensor(n, m, p):
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify(U, V, W, T, n, m, p, tol=1e-9):
    """Vectorized verification of the tensor decomposition."""
    U_r = np.array(U).reshape(n, m, -1)
    V_r = np.array(V).reshape(m, p, -1)
    W_r = np.array(W).reshape(p, n, -1)
    R = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return bool(np.max(np.abs(R - T)) <= tol)

def _schoolbook(n, m, p):
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

def _strassen_2x2x2():
    """
    Rank-7 Strassen decomposition for 2x2 x 2x2.
    Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    U index: i*2+j, V index: j*2+k, W index: k*2+i
    """
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    # Strassen products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0] = 1.0; U[3][0] = 1.0  # a11, a22
    V[0][0] = 1.0; V[3][0] = 1.0  # b11, b22
    # c11 += M1, c22 += M1
    W[0][0] = 1.0; W[3][0] = 1.0  # c11, c22
    
    # M2 = (a21+a22)b11
    U[2][1] = 1.0; U[3][1] = 1.0  # a21, a22
    V[0][1] = 1.0  # b11
    # c21 += M2, c22 -= M2
    W[2][1] = 1.0; W[3][1] = -1.0  # c21, c22
    
    # M3 = a11(b12-b22)
    U[0][2] = 1.0  # a11
    V[1][2] = 1.0; V[3][2] = -1.0  # b12, b22
    # c12 += M3, c22 += M3
    W[1][2] = 1.0; W[3][2] = 1.0  # c12, c22
    
    # M4 = a22(b21-b11)
    U[3][3] = 1.0  # a22
    V[2][3] = 1.0; V[0][3] = -1.0  # b21, b11
    # c11 += M4, c21 += M4
    W[0][3] = 1.0; W[2][3] = 1.0  # c11, c21
    
    # M5 = (a11+a12)b22
    U[0][4] = 1.0; U[1][4] = 1.0  # a11, a12
    V[3][4] = 1.0  # b22
    # c11 -= M5, c12 += M5
    W[0][4] = -1.0; W[1][4] = 1.0  # c11, c12
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1.0; U[0][5] = -1.0  # a21, a11
    V[0][5] = 1.0; V[1][5] = 1.0  # b11, b12
    # c22 += M6
    W[3][5] = 1.0  # c22
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1.0; U[3][6] = -1.0  # a12, a22
    V[2][6] = 1.0; V[3][6] = 1.0  # b21, b22
    # c11 += M7
    W[0][6] = 1.0  # c11
    
    return U, V, W

def _build_strassen_block(n, m, p, b, col_offset):
    """
    Build Strassen decomposition for block b (b in {0,1}) of 2x2 x 2x2.
    Block b corresponds to A[:, 2b:2b+2] @ B[2b:2b+2, :2]
    Returns (U_block, V_block, W_block) each with 7 columns.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U_block = [[0.0] * 7 for _ in range(n*m)]
    V_block = [[0.0] * 7 for _ in range(m*p)]
    W_block = [[0.0] * 7 for _ in range(p*n)]
    
    # Map Strassen indices to full matrix indices
    # A: i in 0..1, j in 0..1 -> full index (i, j+2b)
    # B: j in 0..1, k in 0..1 -> full index (j+2b, k)
    # C: k in 0..1, i in 0..1 -> full index (k, i)
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j][c] != 0:
                    U_block[i*m + (j+2*b)][c] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k][c] != 0:
                    V_block[(j+2*b)*p + k][c] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i][c] != 0:
                    W_block[k*n + i][c] = W_s[k*2+i][c]
    
    return U_block, V_block, W_block

def _build_strassen_2x4x7(n, m, p):
    """
    Build a decomposition using Strassen for the 2x2 blocks.
    Split A (2x4) = [A0 | A1], B (4x7) = [B0; B1]
    C = A0@B0 + A1@B1
    Each Ai (2x2) @ Bi (2x7) needs decomposition.
    
    For 2x2 x 2x7: Split B_i into [B_i1 | B_i2 | B_i3] where B_i1 is 2x2, B_i2 is 2x2, B_i3 is 2x3.
    A_i @ B_i1: rank 7 (Strassen)
    A_i @ B_i2: rank 7 (Strassen) 
    A_i @ B_i3: rank 12 (schoolbook for 2x2x2x3)
    
    Total per block: 7 + 7 + 12 = 26
    Two blocks: 52
    """
    U = [[0.0] * 0 for _ in range(n*m)]
    V = [[0.0] * 0 for _ in range(m*p)]
    W = [[0.0] * 0 for _ in range(p*n)]
    
    def add_terms(U_src, V_src, W_src):
        rank_src = len(U_src[0])
        for c in range(rank_src):
            for row in range(len(U)):
                U[row].append(0)
            for row in range(len(V)):
                V[row].append(0)
            for row in range(len(W)):
                W[row].append(0)
            col = len(U[0]) - 1
            for row in range(len(U_src)):
                if U_src[row][c] != 0:
                    U[row][col] = U_src[row][c]
            for row in range(len(V_src)):
                if V_src[row][c] != 0:
                    V[row][col] = V_src[row][c]
            for row in range(len(W_src)):
                if W_src[row][c] != 0:
                    W[row][col] = W_src[row][c]
    
    for b in range(2):
        # Block b: A[:, 2b:2b+2] @ B[2b:2b+2, :]
        # Split B[2b:2b+2, :] into three parts: cols 0-1, 2-3, 4-6
        # Part 1: 2x2 x 2x2 -> Strassen rank 7
        U1, V1, W1 = _build_strassen_block(n, m, p, b, 0)
        add_terms(U1, V1, W1)
        
        # Part 2: 2x2 x 2x2 (cols 2-3) -> Strassen rank 7
        # Need to shift B columns by 2
        U2, V2, W2 = _build_strassen_block(n, m, p, b, 0)
        # Shift V columns: B rows 2b, 2b+1, cols 2,3
        V2_shifted = [[0.0] * 7 for _ in range(m*p)]
        for j in range(2):
            for k in range(2):
                V2_shifted[(j+2*b)*p + (k+2)][0] = 0  # placeholder
        # Actually, let me rebuild properly
        U2, V2, W2 = _build_strassen_block_shifted(n, m, p, b, 2)
        add_terms(U2, V2, W2)
        
        # Part 3: 2x2 x 2x3 (cols 4-6) -> schoolbook rank 12
        U3, V3, W3 = _build_schoolbook_2x2x2x3(n, m, p, b, 4)
        add_terms(U3, V3, W3)
    
    return {"U": U, "V": V, "W": W}

def _build_strassen_block_shifted(n, m, p, b, k_offset):
    """
    Build Strassen for A[:, 2b:2b+2] @ B[2b:2b+2, k_offset:k_offset+2]
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U_block = [[0.0] * 7 for _ in range(n*m)]
    V_block = [[0.0] * 7 for _ in range(m*p)]
    W_block = [[0.0] * 7 for _ in range(p*n)]
    
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j][c] != 0:
                    U_block[i*m + (j+2*b)][c] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k][c] != 0:
                    V_block[(j+2*b)*p + (k+k_offset)][c] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i][c] != 0:
                    W_block[k*n + i][c] = W_s[k*2+i][c]
    
    return U_block, V_block, W_block

def _build_schoolbook_2x2x2x3(n, m, p, b, k_offset):
    """
    Schoolbook for A[:, 2b:2b+2] @ B[2b:2b+2, k_offset:k_offset+3]
    A is 2x2, B is 2x3, result is 2x3. Rank = 2*2*3 = 12.
    """
    rank = 12
    U_block = [[0.0] * rank for _ in range(n*m)]
    V_block = [[0.0] * rank for _ in range(m*p)]
    W_block = [[0.0] * rank for _ in range(p*n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_block[i*m + (j+2*b)][col] = 1.0
                V_block[(j+2*b)*p + (k+k_offset)][col] = 1.0
                W_block[(k+k_offset)*n + i][col] = 1.0
                col += 1
    
    return U_block, V_block, W_block

def _build_hybrid_decomposition(n, m, p):
    """
    Hybrid: Strassen for 2x2x2 blocks, schoolbook for remainder.
    A (2x4) = [A0 | A1], B (4x7) = [B0; B1]
    C = A0@B0 + A1@B1
    
    For each block Ai (2x2) @ Bi (2x7):
    Split Bi into [Bi1 | Bi2 | Bi3] where Bi1, Bi2 are 2x2, Bi3 is 2x3.
    Ai @ Bi1: Strassen rank 7
    Ai @ Bi2: Strassen rank 7
    Ai @ Bi3: Schoolbook rank 12
    Total per block: 26
    Two blocks: 52
    """
    U = [[0.0] * 0 for _ in range(n*m)]
    V = [[0.0] * 0 for _ in range(m*p)]
    W = [[0.0] * 0 for _ in range(p*n)]
    
    def add_terms(U_src, V_src, W_src):
        rank_src = len(U_src[0])
        for c in range(rank_src):
            for row in range(len(U)):
                U[row].append(0)
            for row in range(len(V)):
                V[row].append(0)
            for row in range(len(W)):
                W[row].append(0)
            col = len(U[0]) - 1
            for row in range(len(U_src)):
                if U_src[row][c] != 0:
                    U[row][col] = U_src[row][c]
            for row in range(len(V_src)):
                if V_src[row][c] != 0:
                    V[row][col] = V_src[row][c]
            for row in range(len(W_src)):
                if W_src[row][c] != 0:
                    W[row][col] = W_src[row][c]
    
    for b in range(2):
        # Part 1: cols 0-1, Strassen
        U1, V1, W1 = _build_strassen_block(n, m, p, b, 0)
        add_terms(U1, V1, W1)
        
        # Part 2: cols 2-3, Strassen
        U2, V2, W2 = _build_strassen_block_shifted(n, m, p, b, 2)
        add_terms(U2, V2, W2)
        
        # Part 3: cols 4-6, Schoolbook
        U3, V3, W3 = _build_schoolbook_2x2x2x3(n, m, p, b, 4)
        add_terms(U3, V3, W3)
    
    return {"U": U, "V": V, "W": W}

def _search_lower(n, m, p, T, start_rank, deadline_check, max_attempts=500):
    """Random search for lower-rank decomposition."""
    rng = random.Random(20240601)
    best = None
    best_rank = start_rank
    for r in range(min(50, start_rank - 1), 24, -1):
        if not deadline_check():
            break
        found = None
        attempts = 0
        while deadline_check() and attempts < max_attempts:
            attempts += 1
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                U[:, c] = rng.choices([-1, 0, 1], k=n * m)
                V[:, c] = rng.choices([-1, 0, 1], k=m * p)
                W[:, c] = rng.choices([-1, 0, 1], k=p * n)
            if _verify(U, V, W, T, n, m, p, 1e-9):
                found = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        if found is not None:
            best = found
            best_rank = r
            break
    return best, best_rank

def construct():
    start = time.time()
    def deadline_check():
        return (time.time() - start) < (150.0 - 10.0)
    
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try hybrid Strassen + schoolbook decomposition (rank 52)
    if deadline_check():
        hybrid = _build_hybrid_decomposition(n, m, p)
        if _verify(hybrid["U"], hybrid["V"], hybrid["W"], T, n, m, p, 1e-9):
            best = hybrid
            best_rank = len(hybrid["U"][0])
    
    # Try random search for even lower rank
    if deadline_check() and best_rank > 30:
        res, r = _search_lower(n, m, p, T, best_rank, deadline_check, max_attempts=200)
        if res is not None and _verify(res["U"], res["V"], res["W"], T, n, m, p, 1e-9):
            best = res
            best_rank = r
    
    return best
# EVOLVE-BLOCK-END
