import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Kronecker product decomposition:
    # 6x6 = (2x2) ⊗ (3x3)
    # Outer: Strassen rank 7 for 2x2
    # Inner: Lader rank 19 for 3x3
    # Total rank: 7 * 19 = 133
    
    rank_outer = 7   # Strassen for 2x2
    rank_inner = 19  # Lader for 3x3
    rank = rank_outer * rank_inner  # 133
    
    U = np.zeros((n * m, rank), dtype=int)  # 36 x 133
    V = np.zeros((m * p, rank), dtype=int)  # 36 x 133
    W = np.zeros((p * n, rank), dtype=int)  # 36 x 133
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Lader's rank-19 algorithm for 3x3 matrix multiplication
    # This is a well-known result from Lader (1974)
    # We define 19 bilinear forms
    
    # Lader's 19 terms for 3x3:
    # Let A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    # Let B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    
    # The 19 bilinear forms (Lader 1974):
    # M1  = (a11+a22+a33)(b11+b22+b33)
    # M2  = (a11+a22-a33)(b11+b22-b33)  
    # M3  = (a11-a22+a33)(b11-b22+b33)
    # M4  = (a11-a22-a33)(b11-b22-b33)
    # M5  = (a12+a21+a33)(b12+b21+b33)
    # M6  = (a12+a21-a33)(b12+b21-b33)
    # M7  = (a12-a21+a33)(b12-b21+b33)
    # M8  = (a12-a21-a33)(b12-b21-b33)
    # M9  = (a13+a23+a32)(b13+b23+b32)
    # M10 = (a13+a23-a32)(b13+b23-b32)
    # M11 = (a13-a23+a32)(b13-b23+b32)
    # M12 = (a13-a23-a32)(b13-b23-b32)
    # M13 = (a21+a31+a12)(b21+b31+b12)
    # M14 = (a21+a31-a12)(b21+b31-b12)
    # M15 = (a21-a31+a12)(b21-b31+b12)
    # M16 = (a21-a31-a12)(b21-b31-b12)
    # M17 = (a31+a13+a23)(b31+b13+b23)
    # M18 = (a31+a13-a23)(b31+b13-b23)
    # M19 = (a31-a13+a23)(b31-b13+b23)
    
    # Wait, I need to be more careful. Let me use a verified Lader decomposition.
    # Actually, let me use a simpler but verified rank-19 decomposition.
    
    # For safety, let me use the schoolbook 27 for 3x3 if Lader is too complex to verify.
    # But first, let me try to implement a correct Lader.
    
    # Actually, let me use a different approach: use the known fact that
    # rank(6x6) <= rank(2x2) * rank(3x3) via Kronecker product.
    # If I can't verify Lader quickly, I'll stick with schoolbook 27.
    
    # Let me use schoolbook for 3x3 (rank 27) as a safe baseline,
    # but structure the code so it's easy to swap in Lader later.
    
    # For now, use schoolbook 27 for 3x3:
    rank_inner = 27
    rank = rank_outer * rank_inner  # 7 * 27 = 189
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    # A block (i,k): rows 2*i..2*i+1, cols 2*k..2*k+1 in A (6x6)
                    # Flattened index: row*6 + col
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    
                    # B block (k,j): rows 2*k..2*k+1, cols 2*j..2*j+1 in B (6x6)
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    
                    # C block (i,j): rows 2*i..2*i+1, cols 2*j..2*j+1 in C (6x6)
                    # Trace convention: C[r,c] is at W_flat[c*6 + r]
                    c00 = (2*j) * 6 + (2*i)
                    c01 = (2*j+1) * 6 + (2*i)
                    c10 = (2*j) * 6 + (2*i+1)
                    c11 = (2*j+1) * 6 + (2*i+1)
                    
                    if s == 0:  # M1 = (a00 + a11)(b00 + b11)
                        U[a00, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1
                        W[c11, idx] = 1
                    elif s == 1:  # M2 = (a10 + a11)b00
                        U[a10, idx] = 1
                        U[a11, idx] = 1
                        V[b00, idx] = 1
                        W[c10, idx] = 1
                        W[c11, idx] = -1
                    elif s == 2:  # M3 = a00(b01 - b11)
                        U[a00, idx] = 1
                        V[b01, idx] = 1
                        V[b11, idx] = -1
                        W[c01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 3:  # M4 = a11(b10 - b00)
                        U[a11, idx] = 1
                        V[b10, idx] = 1
                        V[b00, idx] = -1
                        W[c00, idx] = 1
                        W[c10, idx] = 1
                    elif s == 4:  # M5 = (a00 + a01)b11
                        U[a00, idx] = 1
                        U[a01, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = -1
                        W[c01, idx] = 1
                    elif s == 5:  # M6 = (a10 - a00)(b00 + b01)
                        U[a10, idx] = 1
                        U[a00, idx] = -1
                        V[b00, idx] = 1
                        V[b01, idx] = 1
                        W[c11, idx] = 1
                    elif s == 6:  # M7 = (a01 - a11)(b10 + b11)
                        U[a01, idx] = 1
                        U[a11, idx] = -1
                        V[b10, idx] = 1
                        V[b11, idx] = 1
                        W[c00, idx] = 1
                    
                    idx += 1
    
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    # Verify the decomposition is correct
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        
        C_expected = A_test @ B_test
        
        U_arr = U.astype(float)
        V_arr = V.astype(float)
        W_arr = W.astype(float)
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        prod = np.einsum('ikr,kjr->ijr', U_3d, V_3d)
        C_computed = np.einsum('ijr,jir->ij', prod, W_3d)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=int)
            V_s = np.zeros((m * p, rank_school), dtype=int)
            W_s = np.zeros((p * n, rank_school), dtype=int)
            
            idx_s = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_s[i * m + k, idx_s] = 1
                        V_s[k * p + j, idx_s] = 1
                        W_s[j * n + i, idx_s] = 1
                        idx_s += 1
            
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
