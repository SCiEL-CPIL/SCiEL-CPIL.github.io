"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition.

Strategy: Pad the 2x4 input to 3x4 and the 4x6 output to 3x6, then use a
rank-30 decomposition for 3x3 x 3x3 matrix multiplication on the padded
3x4 x 4x6 problem, combined with a schoolbook term for the remaining
1x4 x 4x6 slice. Total rank: 30 + 4 = 34.

However, the known rank-30 for 3x3x3 is not standard; Strassen gives 7 for 2x2x2.
For 3x3x3, the best known is 19 (Landsberg) but that's complex. 
A simpler structural improvement: use Strassen on 2x2x2 blocks but with
a different partitioning that reduces rank.

Actually, let me reconsider. The current best is 42 from 2*3*7 Strassen blocks.
The inspiration suggests 30+4=34 using a rank-30 3x3x3 algorithm.
But rank-30 for 3x3x3 is not well-known; the best known is 19.

Let me try a different structural approach: use the fact that 2x4 x 4x6 can be
decomposed as:
- 2x2 x 2x6 (left half of A) 
- 2x2 x 2x6 (right half of A)

Each 2x2 x 2x6 can be done with 3 Strassen 2x2x2 = 7*3 = 21 terms.
Total: 2*21 = 42. That's what we have.

Alternative: Can we share some Strassen terms between the two halves?
If we group differently...

Actually, let me try the padding approach more carefully.
Pad A from 2x4 to 3x4 (add a zero row), pad C from 2x6 to 3x6.
Then C_padded = A_padded * B where A_padded is 3x4, B is 4x6.
We can decompose 3x4 x 4x6 as:
- 3x2 x 2x6 (left 2 cols of A_padded)
- 3x2 x 2x6 (right 2 cols of A_padded)

Each 3x2 x 2x6: we can do this with... hmm, not obviously better.

Let me just implement the clean Strassen-based rank-42 construction with
proper verification and a fallback, matching the inspiration's cleaner code style.
"""
import time

DIMENSIONS = [2, 4, 6]


def construct():
    start_time = time.time()
    DEADLINE = 130

    n, m, p = DIMENSIONS  # n=2, m=4, p=6

    # Strassen 2x2x2 coefficients
    # U (A matrix): index i*2+j, i,j in {0,1}
    # V (B matrix): index j*2+k, j,k in {0,1}  
    # W (C matrix): index k*2+i, k,i in {0,1}
    
    # The 7 Strassen products:
    # P0: (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
    # P1: (A10+A11)*B00      -> C10 += 1, C11 -= 1
    # P2: A00*(B01-B11)      -> C01 += 1, C11 += 1
    # P3: A11*(B10-B00)      -> C00 += 1, C10 += 1
    # P4: (A00+A01)*B11      -> C00 -= 1, C01 += 1
    # P5: (A10-A00)*(B00+B01)-> C11 += 1
    # P6: (A01-A11)*(B10+B11)-> C00 += 1

    U_coeffs = [
        [(0, 1), (3, 1)],   # P0: A00+A11
        [(2, 1), (3, 1)],   # P1: A10+A11
        [(0, 1)],           # P2: A00
        [(3, 1)],           # P3: A11
        [(0, 1), (1, 1)],   # P4: A00+A01
        [(2, 1), (0, -1)],  # P5: A10-A00
        [(1, 1), (3, -1)],  # P6: A01-A11
    ]
    
    V_coeffs = [
        [(0, 1), (3, 1)],   # P0: B00+B11
        [(0, 1)],           # P1: B00
        [(1, 1), (3, -1)],  # P2: B01-B11
        [(2, 1), (0, -1)],  # P3: B10-B00
        [(3, 1)],           # P4: B11
        [(0, 1), (1, 1)],   # P5: B00+B01
        [(2, 1), (3, 1)],   # P6: B10+B11
    ]
    
    W_coeffs = [
        [(0, 1), (3, 1)],   # P0: C00+=1, C11+=1
        [(2, 1), (3, -1)],  # P1: C10+=1, C11-=1
        [(1, 1), (3, 1)],   # P2: C01+=1, C11+=1
        [(0, 1), (2, 1)],   # P3: C00+=1, C10+=1
        [(0, -1), (1, 1)],  # P4: C00-=1, C01+=1
        [(3, 1)],           # P5: C11+=1
        [(0, 1)],           # P6: C00+=1
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # (8, 42)
    V = [[0] * rank for _ in range(m * p)]   # (24, 42)
    W = [[0] * rank for _ in range(p * n)]   # (12, 42)
    
    col = 0
    for a_block in range(2):  # A column block: 0 for j in {0,1}, 1 for j in {2,3}
        for b_col_block in range(3):  # B column block: 0 for k in {0,1}, 1 for {2,3}, 2 for {4,5}
            for s in range(7):  # Strassen term
                # Update U: A[i, j] where j = a_block*2 + j_loc
                for (idx, coeff) in U_coeffs[s]:
                    i_loc, j_loc = divmod(idx, 2)
                    j_global = a_block * 2 + j_loc
                    u_row = i_loc * m + j_global
                    U[u_row][col] += coeff
                
                # Update V: B[j, k] where j = a_block*2 + j_loc, k = b_col_block*2 + k_loc
                for (idx, coeff) in V_coeffs[s]:
                    j_loc, k_loc = divmod(idx, 2)
                    j_global = a_block * 2 + j_loc
                    k_global = b_col_block * 2 + k_loc
                    v_row = j_global * p + k_global
                    V[v_row][col] += coeff
                
                # Update W: C[k, i] where k = b_col_block*2 + k_loc, i = i_loc
                for (idx, coeff) in W_coeffs[s]:
                    k_loc, i_loc = divmod(idx, 2)
                    k_global = b_col_block * 2 + k_loc
                    w_row = k_global * n + i_loc
                    W[w_row][col] += coeff
                
                col += 1
    
    assert col == 42, f"Expected 42 factors, got {col}"
    
    # Verify correctness
    def check_tensor_identity(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1 if (j == jp and k == kp and i == ip) else 0
                                actual = 0
                                for r in range(rank):
                                    actual += U[i * m + j][r] * V[jp * p + k][r] * W[kp * n + ip][r]
                                if expected != actual:
                                    return False
        return True
    
    ok = check_tensor_identity(U, V, W, n, m, p)
    if not ok:
        # Fall back to schoolbook rank 48
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Safety fallback
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
