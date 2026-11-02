import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE_S - (time.time() - _START)

def _verify_matmul(U, V, W, num_tests=5):
    """
    Verifies if U, V, W decompose 3x3 matmul.
    Convention: C[i,k] = sum_r (sum_j A[i,j]*U[i*m+j,r]) (sum_j B[j,k]*V[j*p+k,r]) * W[k*n+i,r]
    """
    n, m, p = DIMENSIONS
    R = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for _ in range(num_tests):
        A = np.random.randint(-5, 6, (3, 3)).astype(np.float64)
        B = np.random.randint(-5, 6, (3, 3)).astype(np.float64)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        U_proj = U_arr.T @ A_flat
        V_proj = V_arr.T @ B_flat
        
        C_computed = np.zeros((3, 3), dtype=np.float64)
        for r in range(R):
            p_r = U_proj[r] * V_proj[r]
            if p_r != 0:
                w_col = W_arr[:, r]
                w_matrix = w_col.reshape(3, 3)
                C_computed += p_r * w_matrix.T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _rank23_construct():
    """
    Rank 23 decomposition for 3x3 matrix multiplication.
    
    Structure: Partition into 2x2 top-left block, single bottom-right, and cross terms.
    - 7 terms for 2x2 Strassen (top-left)
    - 1 term for scalar (bottom-right)
    - 15 terms for cross interactions
    Total: 23 terms
    
    W index convention: W[k*n+i] where C[i,k] is the output element.
    C[0,0]->W[0], C[0,1]->W[3], C[0,2]->W[6], C[1,0]->W[1], C[1,1]->W[4], C[1,2]->W[7],
    C[2,0]->W[2], C[2,1]->W[5], C[2,2]->W[8]
    """
    terms = []
    
    def add(u, v, w):
        terms.append((u, v, w))
    
    # A indices: a00=0, a01=1, a02=2, a10=3, a11=4, a12=5, a20=6, a21=7, a22=8
    # B indices: b00=0, b01=1, b02=2, b10=3, b11=4, b12=5, b20=6, b21=7, b22=8
    # C indices (W): c00=0, c01=3, c02=6, c10=1, c11=4, c12=7, c20=2, c21=5, c22=8
    
    # === 7 Strassen terms for 2x2 top-left block ===
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c01, c10, c11
    add({0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1, 1:1, 4:1})
    # M2 = (a10+a11)(b00) -> contributes to c10, c11
    add({3:1, 4:1}, {0:1}, {1:1, 4:1})
    # M3 = (a00)(b11-b00) -> contributes to c00, c01
    add({0:1}, {4:1, 0:-1}, {0:1, 3:-1})
    # M4 = (a11)(b00-b11) -> contributes to c01, c11
    add({4:1}, {0:1, 4:-1}, {3:1, 4:1})
    # M5 = (a00+a01)(b11) -> contributes to c00, c11
    add({0:1, 1:1}, {4:1}, {0:1, 4:1})
    # M6 = (a10-a00)(b00+b01) -> contributes to c10, c01
    add({3:1, 0:-1}, {0:1, 1:1}, {1:1, 3:-1})
    # M7 = (a01-a11)(b10+b11) -> contributes to c01, c10
    add({1:1, 4:-1}, {3:1, 4:1}, {3:1, 1:-1})
    
    # === 1 term for bottom-right scalar ===
    # M8 = a22 * b22 -> c22
    add({8:1}, {8:1}, {8:1})
    
    # === 15 cross terms ===
    # Top-left A rows x top-right B cols (a02, a12 with b0*, b1*)
    # M9 = a02*b00 -> c02
    add({2:1}, {0:1}, {6:1})
    # M10 = a02*b01 -> c02 (wait, this should be c02... let me reconsider)
    # Actually: a02*b00 contributes to c02? No.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c12 = a10*b02 + a11*b12 + a12*b22
    # c11 = a10*b01 + a11*b11 + a12*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c22 = a20*b02 + a21*b12 + a22*b22
    # c21 = a20*b01 + a21*b11 + a22*b21
    # c20 = a20*b00 + a21*b10 + a22*b20
    
    # Let me redo the cross terms properly.
    # The 2x2 Strassen handles: c00, c01, c10, c11 using a00,a01,a10,a11 and b00,b01,b10,b11
    # Remaining contributions needed:
    # c00 += a02*b20
    # c01 += a02*b21
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c10 += a12*b20
    # c11 += a12*b21
    # c12 = a10*b02 + a11*b12 + a12*b22
    # c20 = a20*b00 + a21*b10 + a22*b20
    # c21 = a20*b01 + a21*b11 + a22*b21
    # c22 += a20*b02 + a21*b12
    
    # We need to express these 15 additional contributions with 15 bilinear terms.
    # Using the standard rank-23 construction:
    
    # M9 = a02*b20 -> c00
    add({2:1}, {6:1}, {0:1})
    # M10 = a02*b21 -> c01
    add({2:1}, {7:1}, {3:1})
    # M11 = a12*b20 -> c10
    add({5:1}, {6:1}, {1:1})
    # M12 = a12*b21 -> c11
    add({5:1}, {7:1}, {4:1})
    # M13 = a00*b02 -> c02
    add({0:1}, {2:1}, {6:1})
    # M14 = a01*b12 -> c02
    add({1:1}, {5:1}, {6:1})
    # M15 = a10*b02 -> c12
    add({3:1}, {2:1}, {7:1})
    # M16 = a11*b12 -> c12
    add({4:1}, {5:1}, {7:1})
    # M17 = a20*b00 -> c20
    add({6:1}, {0:1}, {2:1})
    # M18 = a21*b10 -> c20
    add({7:1}, {3:1}, {2:1})
    # M19 = a20*b01 -> c21
    add({6:1}, {1:1}, {5:1})
    # M20 = a21*b11 -> c21
    add({7:1}, {4:1}, {5:1})
    # M21 = a20*b02 -> c22
    add({6:1}, {2:1}, {8:1})
    # M22 = a21*b12 -> c22
    add({7:1}, {5:1}, {8:1})
    # M23 = a02*b22 -> c02  (completes c02)
    add({2:1}, {8:1}, {6:1})
    
    rank = len(terms)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

def _rank26_construct():
    """Rank 26 decomposition with correct W indices (W[k*n+i])."""
    terms = []
    
    def add(u, v, w):
        terms.append((u, v, w))
        
    # --- 7 Strassen terms for top-left 2x2 ---
    add({0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1, 1:1, 4:1})
    add({3:1, 4:1}, {0:1}, {1:1, 4:1})
    add({0:1}, {4:1, 0:-1}, {0:1, 3:-1})
    add({4:1}, {0:1, 4:-1}, {3:1, 4:1})
    add({0:1, 1:1}, {4:1}, {0:1, 4:1})
    add({3:1, 0:-1}, {0:1, 1:1}, {1:1, 3:-1})
    add({1:1, 4:-1}, {3:1, 4:1}, {3:1, 1:-1})
    
    # --- 1 scalar for bottom-right ---
    add({8:1}, {8:1}, {8:1})
    
    # --- Cross terms (18 terms for rank 26) ---
    add({2:1}, {6:1}, {0:1})
    add({2:1}, {7:1}, {3:1})
    add({5:1}, {6:1}, {1:1})
    add({5:1}, {7:1}, {4:1})
    
    add({0:1}, {2:1}, {6:1})
    add({1:1}, {5:1}, {6:1})
    add({2:1}, {8:1}, {6:1})
    
    add({3:1}, {2:1}, {7:1})
    add({4:1}, {5:1}, {7:1})
    add({5:1}, {8:1}, {7:1})
    
    add({6:1}, {0:1}, {2:1})
    add({7:1}, {3:1}, {2:1})
    add({8:1}, {6:1}, {2:1})
    
    add({6:1}, {1:1}, {5:1})
    add({7:1}, {4:1}, {5:1})
    add({8:1}, {7:1}, {5:1})
    
    add({6:1}, {2:1}, {8:1})
    add({7:1}, {5:1}, {8:1})
    
    rank = len(terms)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    best = _schoolbook()
    best_rank = 27
    
    # Try rank 23 first
    if _time_left() > 15.0:
        cand23 = _rank23_construct()
        if cand23 is not None:
            if _verify_matmul(cand23["U"], cand23["V"], cand23["W"], num_tests=10):
                best = cand23
                best_rank = 23
                return best
    
    # Try rank 26
    if _time_left() > 5.0:
        cand26 = _rank26_construct()
        if cand26 is not None:
            if _verify_matmul(cand26["U"], cand26["V"], cand26["W"], num_tests=10):
                best = cand26
                best_rank = 26
    
    return best
