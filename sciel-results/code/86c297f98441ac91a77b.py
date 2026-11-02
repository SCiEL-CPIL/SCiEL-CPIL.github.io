"""Strassen block decomposition for 2x4x7 matrix multiplication tensor."""
import time

DIMENSIONS = [2, 4, 7]

# Strassen 2x2x2 factors (7 multiplications)
# Each factor is a triple of 2x2 matrices (U, V, W)
# Convention: C = U * V * W (conceptually, but we flatten into single index)
# The tensor identity is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k} for 2x2 identity?
# No, the matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k in {0,1}
# Wait, the tensor for 2x2x2 multiplication has entries T[i,j,k] = 1.
# The decomposition is sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k.
# Actually, the standard Strassen decomposition gives:
# C_{0,0} = M1 + M4 - M5 - M7
# C_{0,1} = M3 + M5
# C_{1,0} = M2 + M4
# C_{1,1} = M1 - M3 + M6 + M7
# where M1 = (A00+A11)(B00+B11), M2 = (A10+A11)B00, M3 = A00(B10-B11), 
# M4 = A11(B01-B10), M5 = (A00+A10)B11, M6 = (A01-A11)(B00+B10), M7 = (A10-A00)(B01+B11)

# Let's define the 7 Strassen factors explicitly as (u, v, w) vectors of length 4
# u[i] corresponds to A[i//2, i%2], v[j] to B[j//2, j%2], w[k] to C[k//2, k%2]

# M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
# u: [1,0,0,1], v: [1,0,0,1], w: [1,0,0,1]
# M2 = (A10+A11)B00 -> C10 += 1
# u: [0,0,1,1], v: [1,0,0,0], w: [0,0,1,0]
# M3 = A00(B10-B11) -> C01 += 1, C11 -= 1
# u: [1,0,0,0], v: [0,1,0,-1], w: [0,1,0,-1]
# M4 = A11(B01-B10) -> C10 += 1, C00 -= 1
# u: [0,0,0,1], v: [0,1,-1,0], w: [-1,0,1,0]
# M5 = (A00+A10)B11 -> C01 += 1, C11 += 1
# u: [1,0,1,0], v: [0,0,0,1], w: [0,1,0,1]
# M6 = (A01-A11)(B00+B10) -> C11 += 1
# u: [0,1,0,-1], v: [1,0,1,0], w: [0,0,0,1]
# M7 = (A10-A00)(B01+B11) -> C00 -= 1, C11 += 1
# u: [-1,0,1,0], v: [0,1,0,1], w: [-1,0,0,1]

STRASSEN_FACTORS = [
    ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
    ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2
    ([1,0,0,0], [0,1,0,-1], [0,1,0,-1]),  # M3
    ([0,0,0,1], [0,1,-1,0], [-1,0,1,0]),  # M4
    ([1,0,1,0], [0,0,0,1], [0,1,0,1]),  # M5
    ([0,1,0,-1], [1,0,1,0], [0,0,0,1]),  # M6
    ([-1,0,1,0], [0,1,0,1], [-1,0,0,1]),  # M7
]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Rank: 2 * (3*7 + 4) = 2 * 25 = 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    col = 0
    
    # We have A (2x4), B (4x7), C = A*B (2x7)
    # Split A into A1 (rows 0), A2 (rows 1) - each 1x4? No.
    # Better: Split A into A_top (row 0) and A_bot (row 1), each 1x4.
    # But Strassen works on 2x2. Let's split differently.
    # 
    # Actually, let's split B into left part (cols 0-3, 4x4) and right part (cols 4-6, 4x3).
    # For the 4x4 part: split into four 2x2 blocks. A is 2x4, so split A into A_left (2x2) and A_right (2x2).
    # A_left * B_topleft (2x2) -> use Strassen (7 factors)
    # A_left * B_topright (2x2) -> use Strassen (7 factors)
    # A_right * B_bottomleft (2x2) -> use Strassen (7 factors)
    # A_right * B_bottomright (2x2) -> use Strassen (7 factors)
    # That's 28 factors for the 4x4 part.
    # For the 4x3 part: split B into 4x2 and 4x1.
    # A (2x4) * B_4x2 (4x2): split A into A_left, A_right (each 2x2).
    # A_left * B_4x2_top (2x2) -> Strassen (7)
    # A_right * B_4x2_bot (2x2) -> Strassen (7)
    # That's 14 factors.
    # A (2x4) * B_4x1 (4x1): schoolbook, 8 factors.
    # Total: 28 + 14 + 8 = 50. 
    
    # Helper to add a Strassen factor for sub-blocks
    # A_sub is at rows [a_i, a_i+1], cols [a_j, a_j+1]
    # B_sub is at rows [b_i, b_i+1], cols [b_j, b_j+1]
    # C_sub is at row c_i, cols [c_j, c_j+1]
    
    def add_strassen(a_i, a_j, b_i, b_j, c_i, c_j):
        nonlocal col
        for u, v, w in STRASSEN_FACTORS:
            for di in range(2):
                for dj in range(2):
                    U[a_i * m + a_j + di * 0 + (di if False else 0)][col]  # This is getting complicated
            # Let me redo this with proper index mapping
            pass
    
    # Let me be more careful with indices.
    # A is 2x4. Index (i, j) -> row i*m + j = i*4 + j
    # B is 4x7. Index (j, k) -> row j*p + k = j*7 + k
    # C is 2x7. Index (i, k) -> row k*n + i = k*2 + i (per the trace convention W[k*n+i, r])
    
    # For a 2x2x2 Strassen factor:
    # A sub-block: rows a_i, a_i+1; cols a_j, a_j+1
    # B sub-block: rows b_i, b_i+1; cols b_j, b_j+1
    # C sub-block: row c_i; cols c_j, c_j+1
    # 
    # u[di] (di=0,1) corresponds to A row a_i+di
    # u[dj] (dj=0,1) corresponds to A col a_j+dj
    # So u index di*2+dj corresponds to U[(a_i+di)*m + (a_j+dj), col]
    # 
    # v[di] (di=0,1) corresponds to B row b_i+di
    # v[dj] (dj=0,1) corresponds to B col b_j+dj
    # So v index di*2+dj corresponds to V[(b_i+di)*p + (b_j+dj), col]
    # 
    # w[di] (di=0,1) corresponds to C row c_i (only one row! since n=2 but we're using 2x2x2 for a single output row pair?)
    
    # Wait, I'm confusing myself. The output C is 2x7. For a 2x2x2 multiplication, we get a 2x2 output.
    # But here, each Strassen factor computes a 2x2 block of C.
    # So for A (2x2) * B (2x2) -> C (2x2).
    # A rows: a_i, a_i+1 (but A only has 2 rows, so a_i must be 0)
    # A cols: a_j, a_j+1
    # B rows: b_i, b_i+1
    # B cols: b_j, b_j+1
    # C rows: c_i, c_i+1 (but C only has 2 rows, so c_i must be 0)
    # C cols: c_j, c_j+1
    
    # So a_i = 0, c_i = 0 always.
    # u[di*2+dj] -> U[0*4 + a_j + dj] = U[a_j + dj] for row di? No.
    # u is a vector of length 4. u[0]=A[0,a_j], u[1]=A[0,a_j+1], u[2]=A[1,a_j], u[3]=A[1,a_j+1]
    # So u[d] where d = di*2+dj: U[di*4 + a_j + dj, col]
    
    # v is a vector of length 4. v[0]=B[b_i,b_j], v[1]=B[b_i,b_j+1], v[2]=B[b_i+1,b_j], v[3]=B[b_i+1,b_j+1]
    # So v[d] where d = di*2+dj: V[(b_i+di)*7 + (b_j+dj), col]
    
    # w is a vector of length 4. w[0]=C[0,c_j], w[1]=C[0,c_j+1], w[2]=C[1,c_j], w[3]=C[1,c_j+1]
    # But W index is k*n + i = k*2 + i.
    # C[0,c_j] -> W[c_j*2 + 0] = W[2*c_j]
    # C[0,c_j+1] -> W[(c_j+1)*2 + 0] = W[2*c_j+2]
    # C[1,c_j] -> W[c_j*2 + 1] = W[2*c_j+1]
    # C[1,c_j+1] -> W[(c_j+1)*2 + 1] = W[2*c_j+3]
    # So w[d] where d = di*2+dj: W[2*(c_j+dj) + di, col]
    
    def add_strassen(a_j, b_i, b_j, c_j):
        nonlocal col
        for u, v, w in STRASSEN_FACTORS:
            for d in range(4):
                di = d // 2
                dj = d % 2
                U[a_j + dj][col] += u[d]  # Wait, this doesn't account for di properly
            pass
    
    # Let me rewrite this more carefully.
    # u index d = di*2 + dj
    # A row = di, A col = a_j + dj
    # U index = di * m + (a_j + dj) = di * 4 + a_j + dj
    
    # v index d = di*2 + dj
    # B row = b_i + di, B col = b_j + dj
    # V index = (b_i + di) * p + (b_j + dj) = (b_i + di) * 7 + b_j + dj
    
    # w index d = di*2 + dj
    # C row = di, C col = c_j + dj
    # W index = (c_j + dj) * n + di = (c_j + dj) * 2 + di
    
    def add_strassen(a_j, b_i, b_j, c_j):
        nonlocal col
        for u, v, w in STRASSEN_FACTORS:
            for d in range(4):
                di = d // 2
                dj = d % 2
                U[di * 4 + a_j + dj][col] = u[d]
                V[(b_i + di) * 7 + b_j + dj][col] = v[d]
                W[(c_j + dj) * 2 + di][col] = w[d]
            col += 1
    
    # For the 4x4 part (B cols 0-3):
    # A_left (cols 0-1) * B_topleft (rows 0-1, cols 0-1)
    add_strassen(0, 0, 0, 0)  # 7 factors
    # A_left (cols 0-1) * B_topright (rows 0-1, cols 2-3)
    add_strassen(0, 0, 2, 2)  # 7 factors
    # A_right (cols 2-3) * B_bottomleft (rows 2-3, cols 0-1)
    add_strassen(2, 2, 0, 0)  # 7 factors
    # A_right (cols 2-3) * B_bottomright (rows 2-3, cols 2-3)
    add_strassen(2, 2, 2, 2)  # 7 factors
    
    # For the 4x2 part (B cols 4-5):
    # A_left (cols 0-1) * B_top (rows 0-1, cols 4-5)
    add_strassen(0, 0, 4, 4)  # 7 factors
    # A_right (cols 2-3) * B_bot (rows 2-3, cols 4-5)
    add_strassen(2, 2, 4, 4)  # 7 factors
    
    # For the 4x1 part (B col 6): schoolbook, 8 factors
    for i in range(2):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 7 + 6][col] = 1
            W[6 * 2 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
