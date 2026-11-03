"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _rank_7_2x2x2():
    """Return the 7 rank-1 tensors for 2x2x2 matrix multiplication (Strassen)."""
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # Strassen's 7 products
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We need U, V, W such that sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = C[i,k]
    # Let's define the 7 columns.
    # Row indices for U (A): 00,01,10,11 -> 0,1,2,3
    # Row indices for V (B): 00,01,10,11 -> 0,1,2,3
    # Row indices for W (C): 00,01,10,11 -> 0,1,2,3 (k*2+i)
    # W[k*2+i]: C00=0, C01=1, C10=2, C11=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[2, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We will use a block decomposition.
    # A is 3x5, B is 5x5.
    # We can split the inner dimension m=5 into 2+3.
    # A = [A_2 | A_3], B = [B_2; B_3].
    # C = A_2 * B_2 + A_3 * B_3.
    
    # Part 1: A_2 (3x2) * B_2 (2x5).
    # We can further split the 3 rows of A_2 into 2+1.
    # A_2 = [A_2_top; A_2_bot] where A_2_top is 2x2 and A_2_bot is 1x2.
    # B_2 is 2x5. Split B_2 into 2x2 and 2x3.
    # B_2 = [B_2_top | B_2_bot] where B_2_top is 2x2 and B_2_bot is 2x3.
    
    # This gets complicated. Let's stick to the 2x2 Strassen on the top-left 2x2x2 block.
    # The rest is schoolbook.
    
    # Indices:
    # A: i in 0..2, j in 0..4.
    # B: j in 0..4, k in 0..4.
    # C: i in 0..2, k in 0..4.
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}. Size 2x2x2. Rank 7.
    # Block 2: Everything else.
    # This is not a direct sum because the indices overlap in j and k.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # We can define the rank-1 tensors such that they only contribute to specific (i,j,k) triples.
    
    # If we use Strassen on the sub-tensor where i in {0,1}, j in {0,1}, k in {0,1},
    # the contributions to C[i,k] for i in {0,1}, k in {0,1} are handled.
    # But wait, Strassen computes C_sub = A_sub * B_sub.
    # The full C is A*B.
    # We can decompose A and B into blocks.
    # A = A_00 (2x2) + A_01 (2x3) + A_10 (1x2) + A_11 (1x3) ? No.
    # A is 3x5.
    # Let's split rows of A: I = {0,1} and {2}.
    # Let's split columns of A (rows of B): J = {0,1} and {2,3,4}.
    # Let's split columns of B: K = {0,1} and {2,3,4}.
    
    # A = [ A_{00} A_{01} ]   where A_{00} is 2x2, A_{01} is 2x3
    #     [ A_{10} A_{11} ]   where A_{10} is 1x2, A_{11} is 1x3
    # B = [ B_{00} B_{01} ]   where B_{00} is 2x2, B_{01} is 2x3
    #     [ B_{10} B_{11} ]   where B_{10} is 3x2, B_{11} is 3x3
    
    # C = A B = [ A_{00} B_{00} + A_{01} B_{10}   A_{00} B_{01} + A_{01} B_{11} ]
    #           [ A_{10} B_{00} + A_{11} B_{10}   A_{10} B_{01} + A_{11} B_{11} ]
    
    # Block C_{00} (2x2): A_{00} B_{00} + A_{01} B_{10}.
    # A_{00} B_{00} is 2x2*2x2. We can use Strassen (Rank 7).
    # A_{01} B_{10} is 2x3 * 3x2. This is a matrix product of size 2x3x2.
    # Rank of 2x3x2 is 6 (schoolbook) or maybe less?
    # For 2x3x2, the schoolbook rank is 6.
    # Can we do better? The tensor is M_{2,3,2}.
    # It is known that R(2,3,2) = 6?
    # Let's assume schoolbook for these cross terms to be safe, or try to find a small rank.
    # Actually, 2x3x2 is small enough that schoolbook (6) is likely optimal or close.
    
    # Let's calculate the rank using schoolbook for all parts, except A_{00}B_{00} which uses Strassen.
    # Rank(A_{00}B_{00}) = 7.
    # Rank(A_{01}B_{10}) = 2*3*2 = 12? No, schoolbook is n*m*p. Here n=2, m=3, p=2. Rank=12.
    # Wait, schoolbook rank for n x m * m x p is n*m*p.
    # So A_{01} (2x3) * B_{10} (3x2) has rank 2*3*2 = 12.
    # A_{00} (2x2) * B_{01} (2x3) has rank 2*2*3 = 12.
    # A_{01} (2x3) * B_{11} (3x3) has rank 2*3*3 = 18.
    # A_{10} (1x2) * B_{00} (2x2) has rank 1*2*2 = 4.
    # A_{10} (1x2) * B_{01} (2x3) has rank 1*2*3 = 6.
    # A_{11} (1x3) * B_{10} (3x2) has rank 1*3*2 = 6.
    # A_{11} (1x3) * B_{11} (3x3) has rank 1*3*3 = 9.
    
    # Total Rank = 7 + 12 + 12 + 18 + 4 + 6 + 6 + 9 = 74.
    # This is 74, which is better than 75!
    
    # Let's verify the block decomposition logic.
    # We are summing rank-1 tensors. Since the blocks partition the (i,j,k) indices?
    # No, the blocks of A and B partition the indices.
    # A_{ij} B_{jk} contributions are disjoint in terms of (i,j,k) triples?
    # Yes, because (i,j,k) uniquely determines which A-block and B-block it belongs to.
    # So we can just sum the ranks.
    
    # We need to construct the U, V, W matrices.
    # We will generate the schoolbook tensors for each block and the Strassen tensor for the 2x2x2 block.
    
    U_total = []
    V_total = []
    W_total = []
    
    # Helper to add schoolbook block
    def add_schoolblock(i_start, i_end, j_start, j_end, k_start, k_end):
        # A is (i_end-i_start) x (j_end-j_start)
        # B is (j_end-j_start) x (k_end-k_start)
        ni = i_end - i_start
        nj = j_end - j_start
        nk = k_end - k_start
        for di in range(ni):
            for dj in range(nj):
                for dk in range(nk):
                    i = i_start + di
                    j = j_start + dj
                    k = k_start + dk
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    
                    U_total.append(u)
                    V_total.append(v)
                    W_total.append(w)
    
    # Helper to add Strassen block for 2x2x2 at (i0, j0, k0)
    def add_strassen_block(i0, j0, k0):
        U_s, V_s, W_s = _rank_7_2x2x2()
        for c in range(7):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            
            for di in range(2):
                for dj in range(2):
                    val = U_s[di * 2 + dj, c]
                    if val != 0:
                        i = i0 + di
                        j = j0 + dj
                        u[i * m + j] += val
                        
            for dj in range(2):
                for dk in range(2):
                    val = V_s[dj * 2 + dk, c]
                    if val != 0:
                        j = j0 + dj
                        k = k0 + dk
                        v[j * p + k] += val
                        
            for dk in range(2):
                for di in range(2):
                    val = W_s[dk * 2 + di, c]
                    if val != 0:
                        k = k0 + dk
                        i = i0 + di
                        w[k * n + i] += val
                        
            U_total.append(u)
            V_total.append(v)
            W_total.append(w)
            
    # Blocks:
    # I: 0,1 vs 2
    # J: 0,1 vs 2,3,4
    # K: 0,1 vs 2,3,4
    
    # A_{00} B_{00}: i in [0,2), j in [0,2), k in [0,2). Strassen.
    add_strassen_block(0, 0, 0)
    
    # A_{01} B_{10}: i in [0,2), j in [2,5), k in [0,2). Schoolbook.
    add_schoolblock(0, 2, 2, 5, 0, 2)
    
    # A_{00} B_{01}: i in [0,2), j in [0,2), k in [2,5). Schoolbook.
    add_schoolblock(0, 2, 0, 2, 2, 5)
    
    # A_{01} B_{11}: i in [0,2), j in [2,5), k in [2,5). Schoolbook.
    add_schoolblock(0, 2, 2, 5, 2, 5)
    
    # A_{10} B_{00}: i in [2,3), j in [0,2), k in [0,2). Schoolbook.
    add_schoolblock(2, 3, 0, 2, 0, 2)
    
    # A_{10} B_{01}: i in [2,3), j in [0,2), k in [2,5). Schoolbook.
    add_schoolblock(2, 3, 0, 2, 2, 5)
    
    # A_{11} B_{10}: i in [2,3), j in [2,5), k in [0,2). Schoolbook.
    add_schoolblock(2, 3, 2, 5, 0, 2)
    
    # A_{11} B_{11}: i in [2,3), j in [2,5), k in [2,5). Schoolbook.
    add_schoolblock(2, 3, 2, 5, 2, 5)
    
    rank = len(U_total)
    
    U = np.array(U_total).T # Shape (n*m, rank)
    V = np.array(V_total).T # Shape (m*p, rank)
    W = np.array(W_total).T # Shape (p*n, rank)
    
    # Convert to list of lists as per interface expectation?
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't specify list vs np array, but the seed uses lists.
    # However, the evaluator likely accepts np arrays or lists.
    # Let's return lists to be safe, or np arrays?
    # Seed: return {"U": U, "V": V, "W": W} where U is list of lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
