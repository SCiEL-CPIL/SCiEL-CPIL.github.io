"""Matrix multiplication 2x5 x 5x5 - rank-minimal tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n*m, m*p, p*n  # 10, 25, 10

def _build_tensor():
    """T[a,b,c] where a=(i,j), b=(j',k), c=(k',i'); T = delta_{j,j'}*delta_{k,k'}."""
    T = np.zeros((NM, MP, PN), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            a = i*m + j
            for jp in range(m):   # j'
                for k in range(p):
                    b = jp*p + k
                    for kp in range(p):  # k'
                        for ip in range(n):  # i'
                            c = kp*n + ip
                            if jp == j and kp == k:
                                T[a, b, c] = 1.0
    return T

TENSOR = _build_tensor()

def _verify_fast(U, V, W, tol=1e-9):
    """Fast vectorized verification: sum_r U[a,r]V[b,r]W[c,r] == T[a,b,c]."""
    R = U.shape[1]
    UV = np.einsum('ar,br->abr', U, V)
    result = np.einsum('abr,cr->abc', UV, W)
    return np.allclose(result, TENSOR, atol=tol)

def _verify_exact(U, V, W):
    """Exact verification using Fractions (for half-integer values)."""
    R = U.shape[1]
    Uq = [[Fraction(int(round(2*x)), 2) for x in row] for row in U]
    Vq = [[Fraction(int(round(2*x)), 2) for x in row] for row in V]
    Wq = [[Fraction(int(round(2*x)), 2) for x in row] for row in W]
    for a in range(NM):
        for b in range(MP):
            prod = [Uq[a][r] * Vq[b][r] for r in range(R)]
            for c in range(PN):
                val = sum(prod[r] * Wq[c][r] for r in range(R))
                if val != Fraction(int(TENSOR[a, b, c]), 1):
                    return False
    return True

def _construct_strassen_hybrid():
    """
    Construct a rank-46 decomposition using Strassen's algorithm on 2x2 blocks.
    
    Partition B (5x5) into blocks based on rows {0,1},{2,3},{4} and cols {0,1},{2,3},{4}.
    A (2x5) is split into columns [0:2], [2:4], [4].
    
    Four 2x2x2 sub-problems use Strassen (rank 7 each = 28).
    Remaining cross terms use schoolbook (rank 18).
    Total: 46.
    """
    rank = 46
    U = np.zeros((NM, rank), dtype=np.float64)
    V = np.zeros((MP, rank), dtype=np.float64)
    W = np.zeros((PN, rank), dtype=np.float64)
    
    col = 0
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols):
        """
        Add Strassen decomposition for A[a_rows, a_cols] * B[b_rows, b_cols].
        a_rows: 2 row indices for A
        a_cols: 2 col indices for A  
        b_rows: 2 row indices for B
        b_cols: 2 col indices for B
        """
        nonlocal col
        
        A00_idx = a_rows[0]*m + a_cols[0]
        A01_idx = a_rows[0]*m + a_cols[1]
        A10_idx = a_rows[1]*m + a_cols[0]
        A11_idx = a_rows[1]*m + a_cols[1]
        
        B00_idx = b_rows[0]*p + b_cols[0]
        B01_idx = b_rows[0]*p + b_cols[1]
        B10_idx = b_rows[1]*p + b_cols[0]
        B11_idx = b_rows[1]*p + b_cols[1]
        
        C00_idx = b_cols[0]*n + a_rows[0]
        C01_idx = b_cols[1]*n + a_rows[0]
        C10_idx = b_cols[0]*n + a_rows[1]
        C11_idx = b_cols[1]*n + a_rows[1]
        
        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        U[A00_idx, col] += 1
        U[A11_idx, col] += 1
        V[B00_idx, col] += 1
        V[B11_idx, col] += 1
        W[C00_idx, col] += 1
        W[C11_idx, col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        # C10 += M2
        U[A10_idx, col] += 1
        U[A11_idx, col] += 1
        V[B00_idx, col] += 1
        W[C10_idx, col] += 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        # C01 += M3
        U[A00_idx, col] += 1
        V[B01_idx, col] += 1
        V[B11_idx, col] -= 1
        W[C01_idx, col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        # C10 += M4, C11 += M4
        U[A11_idx, col] += 1
        V[B10_idx, col] += 1
        V[B00_idx, col] -= 1
        W[C10_idx, col] += 1
        W[C11_idx, col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        # C00 -= M5, C11 -= M5
        U[A00_idx, col] += 1
        U[A01_idx, col] += 1
        V[B11_idx, col] += 1
        W[C00_idx, col] -= 1
        W[C11_idx, col] -= 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        U[A10_idx, col] += 1
        U[A00_idx, col] -= 1
        V[B00_idx, col] += 1
        V[B01_idx, col] += 1
        W[C11_idx, col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7, C11 += M7
        U[A01_idx, col] += 1
        U[A11_idx, col] -= 1
        V[B10_idx, col] += 1
        V[B11_idx, col] += 1
        W[C00_idx, col] += 1
        W[C11_idx, col] += 1
        col += 1
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2]
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4]
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3])
    
    # Block 3: A[:, 2:4] * B[2:4, 0:2]
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1])
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4]
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3])
    
    # Remaining terms (schoolbook, rank 18 total):
    
    # A[:, 0:2] * B[0:2, 4]  (2x2 x 2x1) -> 4 mults
    for i in [0, 1]:
        for j in [0, 1]:
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
    
    # A[:, 2:4] * B[2:4, 4]  (2x2 x 2x1) -> 4 mults
    for i in [0, 1]:
        for j in [2, 3]:
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
    
    # A[:, 4] * B[4, 0:2]    (2x1 x 1x2) -> 4 mults
    for i in [0, 1]:
        for k in [0, 1]:
            U[i*m + 4, col] = 1
            V[4*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # A[:, 4] * B[4, 2:4]    (2x1 x 1x2) -> 4 mults
    for i in [0, 1]:
        for k in [2, 3]:
            U[i*m + 4, col] = 1
            V[4*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # A[:, 4] * B[4, 4]      (2x1 x 1x1) -> 2 mults
    for i in [0, 1]:
        U[i*m + 4, col] = 1
        V[4*p + 4, col] = 1
        W[4*n + i, col] = 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W

def _schoolbook():
    """Schoolbook decomposition, rank 50."""
    rank = n*m*p
    U = np.zeros((NM, rank), dtype=np.float64)
    V = np.zeros((MP, rank), dtype=np.float64)
    W = np.zeros((PN, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _to_list(U):
    """Convert numpy array to list of lists with clean half-integers."""
    def clean(x):
        hx = round(2*x)
        if abs(hx - round(hx)) < 1e-6:
            return int(hx)/2
        return float(x)
    return [[clean(x) for x in row] for row in U]

def construct():
    t0 = time.time()
    DEADLINE = 120.0
    
    # Schoolbook fallback (rank 50)
    U_sch, V_sch, W_sch = _schoolbook()
    best = (U_sch, V_sch, W_sch)
    best_rank = 50
    
    # Try Strassen hybrid (rank 46)
    try:
        if time.time() - t0 < DEADLINE - 5:
            U46, V46, W46 = _construct_strassen_hybrid()
            if _verify_fast(U46, V46, W46, tol=1e-9):
                if _verify_exact(U46, V46, W46):
                    best = (U46, V46, W46)
                    best_rank = 46
    except Exception:
        pass
    
    U, V, W = best
    Uf = _to_list(U)
    Vf = _to_list(V)
    Wf = _to_list(W)
    return {"U": Uf, "V": Vf, "W": Wf}
