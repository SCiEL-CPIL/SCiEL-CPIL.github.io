"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random

def _trivial_decomposition(n, m, p):
    """
    Trivial rank-n*m*p decomposition: one term per (i,j,k) triple.
    U[i*m+j, idx] = 1, V[j*p+k, idx] = 1, W[k*n+i, idx] = 1
    This is always correct.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_vec, v_vec, w_vec) where:
      u_vec is length 4 (indices 0,1 for A[0,0],A[0,1]; 2,3 for A[1,0],A[1,1])
      v_vec is length 4 (indices 0,1 for B[0,0],B[0,1]; 2,3 for B[1,0],B[1,1])
      w_vec is length 4 (indices 0,1 for C[0,0],C[0,1]; 2,3 for C[1,0],C[1,1])
    
    Strassen:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B10)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B10)
      M7 = (A01-A11)(B01+B11)
      
      C00 = M1+M4-M5+M7
      C01 = M3+M5
      C10 = M2+M4
      C11 = M1-M2+M3+M6
    """
    # Index convention: A[i*2+j], B[i*2+j], C[i*2+j]
    terms = []
    # M1 = (A00+A11)(B00+B11)
    u1 = [1,0,0,1]   # A00 + A11
    v1 = [1,0,0,1]   # B00 + B11
    w1 = [1,0,0,1]   # C00 + C11
    terms.append((u1, v1, w1))
    # M2 = (A10+A11)B00
    u2 = [0,0,1,1]   # A10 + A11
    v2 = [1,0,0,0]   # B00
    w2 = [0,0,1,0]   # C10
    terms.append((u2, v2, w2))
    # M3 = A00(B10-B11)
    u3 = [1,0,0,0]   # A00
    v3 = [0,0,1,-1]  # B10 - B11
    w3 = [0,1,0,0]   # C01
    terms.append((u3, v3, w3))
    # M4 = A11(B01-B10)
    u4 = [0,0,0,1]   # A11
    v4 = [0,1,-1,0]  # B01 - B10
    w4 = [1,0,1,0]   # C00 + C10
    terms.append((u4, v4, w4))
    # M5 = (A00+A01)B11
    u5 = [1,1,0,0]   # A00 + A01
    v5 = [0,0,0,1]   # B11
    w5 = [-1,0,0,0]  # -C00
    terms.append((u5, v5, w5))
    # M6 = (A10-A00)(B00+B10)
    u6 = [-1,0,1,0]  # A10 - A00
    v6 = [1,0,1,0]   # B00 + B10
    w6 = [0,0,0,1]   # C11
    terms.append((u6, v6, w6))
    # M7 = (A01-A11)(B01+B11)
    u7 = [0,1,0,-1]  # A01 - A11
    v7 = [0,1,0,1]   # B01 + B11
    w7 = [1,0,0,0]   # C00
    terms.append((u7, v7, w7))
    return terms

def _verify_decomposition(result, n=2, m=4, p=6, trials=50):
    """Verify a decomposition numerically with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def _block_strassen_decomposition(n, m, p):
    """
    Decompose M_{n,m,p} using block structure and Strassen for 2x2x2 subproblems.
    
    For n=2, m=4, p=6:
    Split A (2x4) into [A1 | A2] where A1, A2 are 2x2.
    Split B (4x6) into [B1; B2] where B1, B2 are 2x6.
    C = A1*B1 + A2*B2
    
    Each 2x2 * 2x6 product: split the 6 into three 2x2 blocks.
    2x2 * 2x2 uses Strassen (rank 7).
    Total: 2 (A blocks) * 3 (B blocks) * 7 = 42 terms.
    """
    n, m, p = 2, 4, 6
    strassen_terms = _strassen_2x2x2_terms()
    
    # We'll build U, V, W incrementally
    U_terms = []  # list of (index_in_A, value) per term
    V_terms = []  # list of (index_in_B, value) per term
    W_terms = []  # list of (index_in_C, value) per term
    
    # A is 2x4: A[i][j] for i in 0,1; j in 0,1,2,3
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B is 4x6: B[j][k] for j in 0,1,2,3; k in 0,...,5
    # B1 = B[0:2, :], B2 = B[2:4, :]
    # B1 split into B1a=B[0:2,0:2], B1b=B[0:2,2:4], B1c=B[0:2,4:6]
    # B2 split into B2a=B[2:4,0:2], B2b=B[2:4,2:4], B2c=B[2:4,4:6]
    # C is 2x6: C[i][k]
    # C[:, 0:2] = A1*B1a + A2*B2a
    # C[:, 2:4] = A1*B1b + A2*B2b
    # C[:, 4:6] = A1*B1c + A2*B2c
    
    for a_block in range(2):  # A1 or A2
        for b_block in range(3):  # B1a,B1b,B1c or B2a,B2b,B2c
            for (u_s, v_s, w_s) in strassen_terms:
                # u_s is for a 2x2 A block, v_s for 2x2 B block, w_s for 2x2 C block
                # Map to global indices
                u_global = [0] * (n * m)  # 8 elements
                v_global = [0] * (m * p)  # 24 elements
                w_global = [0] * (p * n)  # 12 elements
                
                # A block: rows 0,1; cols 2*a_block to 2*a_block+1
                for idx in range(4):
                    si, sj = idx // 2, idx % 2  # local row, col in 2x2
                    gi = si  # global row (n=2)
                    gj = 2 * a_block + sj  # global col
                    u_global[gi * m + gj] = u_s[idx]
                
                # B block: rows 2*a_block to 2*a_block+1; cols 2*b_block to 2*b_block+1
                for idx in range(4):
                    si, sj = idx // 2, idx % 2  # local row, col in 2x2
                    gj = 2 * a_block + si  # global row in B
                    gk = 2 * b_block + sj  # global col
                    v_global[gj * p + gk] = v_s[idx]
                
                # C block: rows 0,1; cols 2*b_block to 2*b_block+1
                for idx in range(4):
                    si, sj = idx // 2, idx % 2  # local row, col in 2x2
                    gi = si  # global row
                    gk = 2 * b_block + sj  # global col
                    # W index: k*n + i
                    w_global[gk * n + gi] = w_s[idx]
                
                U_terms.append(u_global)
                V_terms.append(v_global)
                W_terms.append(w_global)
    
    rank = len(U_terms)
    U = [row[:] for row in zip(*U_terms)] if U_terms else []
    V = [row[:] for row in zip(*V_terms)] if V_terms else []
    W = [row[:] for row in zip(*W_terms)] if W_terms else []
    
    # Fix shapes
    U = [[U[idx][col] for col in range(rank)] for idx in range(n * m)]
    V = [[V[idx][col] for col in range(rank)] for idx in range(m * p)]
    W = [[W[idx][col] for col in range(rank)] for idx in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 6 matrix.
    
    Strategy: Use block Strassen decomposition. Split the 4x6 product into
    2x2x2 subproblems and apply Strassen (rank 7) to each.
    Total rank: 2 * 3 * 7 = 42.
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 110
    
    # First try the block Strassen decomposition
    result = _block_strassen_decomposition(2, 4, 6)
    
    if time.time() < deadline - 20:
        if _verify_decomposition(result, 2, 4, 6, trials=50):
            return result
    
    # Fallback to trivial decomposition (always correct)
    result = _trivial_decomposition(2, 4, 6)
    if time.time() < deadline - 5:
        if _verify_decomposition(result, 2, 4, 6, trials=20):
            return result
    
    return result
# EVOLVE-BLOCK-END
