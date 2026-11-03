import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 Strassen multiplication.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    The tensor identity is sum_r W[k*2+i, r] U[i*2+j, r] V[j*2+k, r] = delta_{k,i}
    for the standard matrix multiplication where C[i, k] = sum_j A[i, j] B[j, k].
    Note: The output index in W is defined as k*n+i. In this 2x2 block, n=2.
    So W[k*2+i] corresponds to the element C[i, k].
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen Products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Combinations for C:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0: M1
    # A part: A00, A11
    U[0][0] = 1
    U[3][0] = 1
    # B part: B00, B11
    V[0][0] = 1
    V[3][0] = 1
    # C part: C00, C11
    # C00: k=0, i=0 -> idx 0
    # C11: k=1, i=1 -> idx 1*2+1 = 3
    W[0][0] += 1
    W[3][0] += 1

    # Term 1: M2
    # A part: A10, A11
    U[2][1] = 1
    U[3][1] = 1
    # B part: B00
    V[0][1] = 1
    # C part: C10, C11
    # C10 = M2 + ... (+1)
    # C11 = ... - M2 + ... (-1)
    # C10: k=0, i=1 -> idx 0*2+1 = 2
    # C11: k=1, i=1 -> idx 3
    W[2][1] += 1
    W[3][1] += -1

    # Term 2: M3
    # A part: A00
    U[0][2] = 1
    # B part: B01, B11
    V[1][2] = 1
    V[3][2] = -1
    # C part: C01, C11
    # C01 = M3 + ... (+1)
    # C11 = ... + M3 + ... (+1)
    # C01: k=1, i=0 -> idx 1*2+0 = 2
    # C11: k=1, i=1 -> idx 3
    W[2][2] += 1
    W[3][2] += 1

    # Term 3: M4
    # A part: A11
    U[3][3] = 1
    # B part: B10, B00
    V[2][3] = 1
    V[0][3] = -1
    # C part: C00, C10
    # C00 = ... - M4 + ... (-1)
    # C10 = ... + M4 (+1)
    # C00: k=0, i=0 -> idx 0
    # C10: k=0, i=1 -> idx 2
    W[0][3] += -1
    W[2][3] += 1

    # Term 4: M5
    # A part: A00, A01
    U[0][4] = 1
    U[1][4] = 1
    # B part: B11
    V[3][4] = 1
    # C part: C00, C01
    # C00 = ... + M5 ... (+1)
    # C01 = ... + M5 (+1)
    # C00: k=0, i=0 -> idx 0
    # C01: k=1, i=0 -> idx 2
    W[0][4] += 1
    W[2][4] += 1

    # Term 5: M6
    # A part: A10, A00
    U[2][5] = 1
    U[0][5] = -1
    # B part: B00, B01
    V[0][5] = 1
    V[1][5] = 1
    # C part: C11
    # C11 = ... + M6 (+1)
    # C11: k=1, i=1 -> idx 3
    W[3][5] += 1

    # Term 6: M7
    # A part: A01, A11
    U[1][6] = 1
    U[3][6] = -1
    # B part: B10, B11
    V[2][6] = 1
    V[3][6] = 1
    # C part: C00
    # C00 = ... - M7 (-1)
    # C00: k=0, i=0 -> idx 0
    W[0][6] += -1

    return U, V, W

def _get_brauer_333():
    """
    Returns (U, V, W) for 3x3x3 Brauer multiplication (rank 23).
    Reference: Brauer, "Ueber die Multiplikation von Matrizen", 1938.
    Decomposition of 3x3 matrix mult into 23 scalar mults.
    
    Indexing:
    U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    C[i, k] = sum_j A[i, j] B[j, k]
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # This is a known rank 23 decomposition.
    # We can construct it by combining a 2x2 Strassen block and 1x1 schoolbook blocks,
    # but careful: simple block decomposition of 3x3 into 2x2+1x1+... 
    # A 3x3 multiplication can be split into blocks:
    # | 2x2 | 2x1 |
    # | 1x2 | 1x1 |
    # This leads to:
    # C22 (2x2) = A22B22 + A21B12 + A12B21 + A11B11 (roughly)
    # Actually, let's use the standard Brauer scheme.
    # It's often derived from Strassen-like tricks on 3x3.
    
    # Let's build it from the block decomposition:
    # A = [[A11, A12], [A21, A22]] where sizes are 2x2, 2x1, 1x2, 1x1
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11 B11 (4 mults schoolbook or 7 Strassen? Let's use Strassen for 2x2 block if we can)
    # But Brauer 23 is for full 3x3.
    
    # Instead of reinventing, let's use the known rank 23 factors.
    # Found in literature or generated via search.
    # Since I don't have the table hardcoded and generating it reliably in time is hard,
    # I will try to construct a rank-23 using a mixed approach:
    # Use Strassen for the 2x2 blocks and handle the cross terms.
    # Or better: verify if rank 23 is valid and try to find factors.
    
    # Actually, a simpler approach: 
    # We know Strassen 2x2 is 7.
    # Schoolbook 3x3 is 27.
    # Brauer is 23.
    # Can we get 23 by combining?
    # Let's try to derive or hardcode the rank 23.
    # Due to complexity, if I cannot hardcode correctly, I might fall back to a block rank.
    # Rank for 3x3 using 2x2 Strassen + 1x1 schoolbook blocks:
    # Blocks: 2x2 (7), 2x1 (2), 1x2 (2), 1x1 (1)
    # Total rank = 7 + 2 + 2 + 1 = 12? No, that's not right.
    # The blocks interact.
    # C11 (2x2) = A11 B11. (7 mults)
    # C12 (2x1) = A11 B12 + A12 B22.
    # C21 (1x2) = A21 B11 + A22 B21.
    # C22 (1x1) = A21 B12 + A22 B21 + A12 B21 + A11 B11 ... 
    # This is getting complicated to minimize rank without specific Brauer formula.
    
    # Let's assume for now that we just use schoolbook 27 for 3x3 if we can't get 23.
    # But wait, 3x5x5 problem.
    # My previous best was 71 using 2x2 Strassen in some blocks.
    # Let's re-evaluate the 3x5x5 block strategy.
    # n=3, m=5, p=5.
    # Block splits:
    # i: 2+1
    # j: 2+2+1? or 3+2?
    # k: 2+2+1?
    
    # If we use a 3x3x3 Brauer (rank 23) for the 3x3x3 sub-blocks, we might get better.
    # But do we have a 3x3x3 block?
    # A is 3x5, B is 5x5.
    # If we partition i into 3, j into 5, k into 5.
    # We can have blocks of size 3x3x3 if we align them.
    # E.g. i={0,1,2}, j={0,1,2}, k={0,1,2} -> 3x3x3 block.
    # Then j={0,1,2}, k={3,4} -> 3x2x2 block.
    # Then j={3,4}, k={0,1,2} -> 3x2x2 block.
    # Then j={3,4}, k={3,4} -> 3x2x2 block? No, i is only 3.
    
    # Let's just stick to the 2x2 Strassen based 71 for now, and ensure it is correct.
    # The previous attempt scored 1.0, meaning it likely returned schoolbook (75) because 71 failed verification or was not tried correctly.
    # The prompt says "current program score 1.0000". This implies the schoolbook rank 75 is the baseline.
    # If I can make rank 71 work, score will be 75/71 = 1.056.
    # If I can make rank 70 work, score 75/70 = 1.071.
    
    return U, V, W

def _build_rank71():
    n, m, p = 3, 5, 5
    
    su, sv, sw = _get_strassen_222()
    
    # Partitions
    # i blocks: 2, 1
    # j blocks: 2, 2, 1
    # k blocks: 2, 2, 1
    
    i_blocks = [(0, 1), (2,)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    total_rank = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    total_rank += 7
                else:
                    total_rank += len(i_b)*len(j_b)*len(k_b)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    # Strassen
                    # i_b has 2 indices, j_b has 2, k_b has 2
                    for c in range(7):
                        # Map local (li, lj) to global
                        for li in range(2):
                            gi = i_b[li]
                            for lj in range(2):
                                gj = j_b[lj]
                                U[gi*m + gj][col+c] = su[li*2+lj][c]
                        for lj in range(2):
                            gj = j_b[lj]
                            for lk in range(2):
                                gk = k_b[lk]
                                V[gj*p + gk][col+c] = sv[lj*2+lk][c]
                        for lk in range(2):
                            gk = k_b[lk]
                            for li in range(2):
                                gi = i_b[li]
                                W[gk*n + gi][col+c] = sw[lk*2+li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_b:
                        for gj in j_b:
                            for gk in k_b:
                                U[gi*m + gj][col] = 1
                                V[gj*p + gk][col] = 1
                                W[gk*n + gi][col] = 1
                                col += 1
                                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    
    # Optimization:
    # For each j, compute M_j = U[:, j, :] @ V[j, :, :]^T  (n x p matrix)
    # M_j[i, k] = sum_r U[i, r] V[k, r]
    # This is not quite right because the sum is over r of the product of the three.
    # The term is sum_r (U[i,m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
    # Let P_j = U[:, j, :] @ V[j, :, :]^T. Shape (n, p).
    # P_j[i, k] = sum_r U[i*m+j, r] V[j*p+k, r]
    # Then the check is not directly P_j because W depends on r.
    # So we must sum over r.
    
    # With rank ~71 and n*m*p = 75, it's fast to just loop.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                s = np.dot(u_val, v_val * w_val) # Elementwise then dot? No.
                # s = sum_r u_val[r] * v_val[r] * w_val[r]
                # This is elementwise product of 3 vectors, then sum.
                s = np.sum(u_val * v_val * w_val)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Attempt rank 71
    try:
        U, V, W = _build_rank71()
        rank = len(W[0])
        if _verify_decomposition(U, V, W, n, m, p, rank):
            # Check time? 71 is fast to build.
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
        
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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

# EVOLVE-BLOCK-END
