import time
import itertools

DIMENSIONS = [2, 4, 7]
n, m, p = DIMENSIONS
RANK_TARGET = 50  # Based on Strassen-like decomposition: 2*(4*7) = 56 schoolbook, but we can do better
# Actually, for 2x4x7, the rank is at most 2*4*7=56.
# A common bound is R(n,m,p) <= R(2n,2m,2p)/something.
# Let's try to construct a rank-50 or lower decomposition.
# Known: R(2,4,7) <= 49? Or maybe 50?
# Let's use the "Strassen-like" approach:
# C = AB where A is 2x4, B is 4x7.
# Split A into A1 (2x2) and A2 (2x2).
# Split B into B1 (2x7) and B2 (2x7).
# C1 = A1*B1, C2 = A1*B2, C3 = A2*B1, C4 = A2*B2.
# This gives C = [C1 C2; C3 C4] which is 2x14, wait no.
# A is 2x4. B is 4x7.
# A = [A1 A2] where A1, A2 are 2x2.
# B = [B1; B2] where B1, B2 are 2x7.
# C = A1*B1 + A2*B2.
# This is a sum of two products of 2x2 and 2x7 matrices.
# R(2,2,7) <= 2*2*7 = 28?
# Actually, R(2,2,7) is at most 2*2*7=28.
# So total rank <= 2 * 28 = 56. No improvement.

# Let's try a different split.
# A is 2x4. B is 4x7.
# We can use the identity:
# C = (A1 + A2) (B1 + B2) - A1 B2 - A2 B1 + A1 B1 + A2 B2 ? No.
# C = A1 B1 + A2 B2.
# Let M1 = (A1 + A2) (B1 + B2) = A1 B1 + A1 B2 + A2 B1 + A2 B2
# Let M2 = A1 B2
# Let M3 = A2 B1
# Then A1 B1 + A2 B2 = M1 - M2 - M3.
# So we need 3 multiplications of size (2x4) x (4x7)? No, the sums change the dimensions.
# (A1+A2) is 2x4. (B1+B2) is 4x7.
# So M1 is 2x7. Cost R(2,4,7).
# M2 = A1 B2. A1 is 2x2, B2 is 2x7. Cost R(2,2,7).
# M3 = A2 B1. A2 is 2x2, B1 is 2x7. Cost R(2,2,7).
# Total rank = R(2,4,7) + 2*R(2,2,7).
# This is recursive. R(2,4,7) = R(2,4,7) + 2*R(2,2,7). This implies R(2,2,7)=0, which is false.
# The logic is flawed because M1 is not a product of the original factors in a simple tensor sense that reduces rank directly without the same dimensions.

# Let's look at the specific dimensions: 2, 4, 7.
# Schoolbook rank = 56.
# Is there a known better rank?
# R(2,4,7) <= R(2,2,2) + R(2,2,2) + R(2,2,2) + ... ?
# Using the "block Strassen" idea:
# Multiply a 2x4 by 4x7.
# Split into four 2x2 blocks in A, and four 2x2 or 2x3 blocks in B?
# B is 4x7. Split into [B11 B12] where B11 is 2x3, B12 is 2x4? No, 7 is odd.
# Split B into B1 (2x3) and B2 (2x4).
# A into A1 (2x2) and A2 (2x2).
# C = [A1 A2] * [B1; B2] = A1 B1 + A2 B2.
# A1 B1 is (2x2)*(2x3) -> 2x3. Rank <= 2*2*3 = 12.
# A2 B2 is (2x2)*(2x4) -> 2x4. Rank <= 2*2*4 = 16.
# Total rank = 12 + 16 = 28?
# Wait, R(2,2,3) is actually 8? (Strassen for 2x2x2 is 7, but 2x2x3?).
# R(2,2,3) <= 8.
# R(2,2,4) <= 14 (2 * R(2,2,2) = 14).
# Total rank = 8 + 14 = 22?
# Let's verify.
# C = A1 B1 + A2 B2.
# The tensor for C is the sum of the tensor for A1 B1 and the tensor for A2 B2.
# They are disjoint in terms of output indices?
# A1 B1 contributes to C[:, 0:3].
# A2 B2 contributes to C[:, 3:7].
# Since the output indices are disjoint, the rank is the sum of the ranks.
# So we need to find the minimal rank for (2x2)*(2x3) and (2x2)*(2x4).
# R(2,2,3):
#   Schoolbook: 12.
#   Can we do better?
#   (2x2)*(2x3) can be seen as two (2x2)*(2x1) multiplications.
#   R(2,2,1) = 4.
#   So R(2,2,3) <= 4 + 4 = 8?
#   Actually, R(2,2,3) is known to be 8?
#   Let's check.
#   A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]].
#   C = [[ae+ch, af+ci, ag+cj], [be+dh, bf+di, bg+dj]].
#   We can compute:
#   M1 = (a+b+c+d)(e+f+h+i) ? No.
#   Standard Strassen for 2x2x2 uses 7 mults.
#   For 2x2x3, we can just do two 2x2x2 mults if we pad B to 2x4? No, that increases rank.
#   Let's try to find a rank 8 decomposition for 2x2x3.
#   Actually, R(2,2,3) = 8 is a known result?
#   Let's assume R(2,2,3) = 8.
#   R(2,2,4) = 2 * R(2,2,2) = 14.
#   Total rank = 8 + 14 = 22.
#   This is much better than 56.

# Let's construct the rank-22 decomposition.
# Part 1: A1 (2x2) * B1 (2x3) -> C1 (2x3). Rank 8.
# Part 2: A2 (2x2) * B2 (2x4) -> C2 (2x4). Rank 14.
# Total U, V, W will have 22 columns.
# U shape: (2*4, 22) = (8, 22)
# V shape: (4*7, 22) = (28, 22)
# W shape: (7*2, 22) = (14, 22)
# Note: W index is k*n + i. k in 0..6, i in 0..1.
# C1 contributes to k=0,1,2.
# C2 contributes to k=3,4,5,6.

# Decomposition for 2x2x3 (Rank 8):
# A = [[a0, a1], [a2, a3]]
# B = [[b0, b1, b2], [b3, b4, b5]]
# C = [[a0b0+a2b3, a0b1+a2b4, a0b2+a2b5], [a1b0+a3b3, a1b1+a3b4, a1b2+a3b5]]
# This is equivalent to multiplying two 2x2 matrices if we split B into B_left (2x2) and B_right (2x1).
# C_left = A * B_left (2x2x2). Rank 7.
# C_right = A * B_right (2x2x1). Rank 4.
# Total rank = 7 + 4 = 11?
# Wait, the output indices are disjoint?
# C_left is C[:, 0:2]. C_right is C[:, 2:3].
# Yes, disjoint. So rank is sum of ranks.
# R(2,2,2) = 7.
# R(2,2,1) = 4.
# Total for 2x2x3 = 7 + 4 = 11.
# Is there a better way?
# R(2,2,3) is actually 8?
# Let's check if R(2,2,3) can be 8.
# If R(2,2,3)=8, then total rank = 8 + 14 = 22.
# If R(2,2,3)=11, then total rank = 11 + 14 = 25.
# Let's assume the sum of ranks is the best we can do easily.
# R(2,2,2) = 7 (Strassen).
# R(2,2,1) = 4 (Schoolbook, since rank of 2x2x1 tensor is 4?
#   T_{i,j,k} = A_{i,l} B_{l,k}.
#   For k fixed, it's a bilinear form.
#   Rank of 2x2x1 is 4?
#   U (2x2, 4), V (2x1, 4), W (1x2, 4).
#   Actually, R(2,2,1) = 4.
# So 2x2x3 rank = 7 + 4 = 11.
# 2x2x4 rank = 7 + 7 = 14.
# Total rank = 11 + 14 = 25.

# Let's try to optimize the 2x2x3 part.
# Is R(2,2,3) = 8 possible?
# If yes, total = 22.
# Let's use the 25 rank decomposition first, as it is easier to construct.
# We can try to improve later.

# Constructing the 25-rank decomposition:
# Block 1: A1 (2x2) * B1 (2x3).
#   B1 = [B11 (2x2), B12 (2x1)].
#   C1 = [C11 (2x2), C12 (2x1)].
#   C11 = A1 * B11. Rank 7.
#   C12 = A1 * B12. Rank 4.
# Block 2: A2 (2x2) * B2 (2x4).
#   B2 = [B21 (2x2), B22 (2x2)].
#   C2 = [C21 (2x2), C22 (2x2)].
#   C21 = A2 * B21. Rank 7.
#   C22 = A2 * B22. Rank 7.
# Total rank = 7 + 4 + 7 + 7 = 25.

# Indices:
# A: 2x4. A1 = A[:, 0:2], A2 = A[:, 2:4].
# B: 4x7. B1 = B[0:2, :3], B2 = B[2:4, 3:7].
# Wait, B is 4x7.
# B1 is rows 0,1 and cols 0,1,2.
# B2 is rows 2,3 and cols 3,4,5,6.
# C = A1*B1 + A2*B2.
# C has 2 rows and 7 cols.
# C[:, 0:3] = A1*B1.
# C[:, 3:7] = A2*B2.
# This matches.

# Strassen for 2x2x2:
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B10-B11)
# M4 = A11(B01-B10)
# M5 = (A00+A10)B11
# M6 = (A01-A10)(B00+B10)
# M7 = (A01-A11)(B01+B11)
# C00 = M1 - M4 + M5
# C01 = M4 + M7
# C10 = M2 + M3
# C11 = M1 - M3 + M6

# Schoolbook for 2x2x1:
# M1 = A00*B0
# M2 = A01*B1
# M3 = A10*B0
# M4 = A11*B1
# C0 = M1 + M2
# C1 = M3 + M4

# We need to map these to the global U, V, W matrices.
# Global U has 8 rows (A indices), 25 cols.
# Global V has 28 rows (B indices), 25 cols.
# Global W has 14 rows (C indices k*n+i), 25 cols.

# Let's build the columns for each multiplication.
# Column 0-6: A1*B11 (2x2x2 Strassen)
# Column 7-10: A1*B12 (2x2x1 Schoolbook)
# Column 11-17: A2*B21 (2x2x2 Strassen)
# Column 18-24: A2*B22 (2x2x2 Strassen)

# Helper to add a column for a 2x2x2 Strassen multiplication.
# Inputs:
#   a_rows: 2 indices in A
#   b_rows: 2 indices in B
#   c_rows: 2 indices in C (k values)
#   col_offset: starting column index
#   sign: 1 or -1 (not needed for positive ranks)

def add_strassen_222(U, V, W, a_rows, b_rows, c_rows, col_offset):
    # a_rows = [a0, a1]
    # b_rows = [b0, b1]
    # c_rows = [c0, c1] (k indices)
    # W index = k*n + i. n=2.
    # c0 -> k0, c1 -> k1.
    
    # M1 = (A00+A11)(B00+B11)
    U[a_rows[0]][col_offset] += 1
    U[a_rows[1]][col_offset] += 1
    V[b_rows[0]][col_offset] += 1
    V[b_rows[1]][col_offset] += 1
    W[c_rows[0]*n + 0][col_offset] += 1 # C00
    W[c_rows[0]*n + 1][col_offset] += 1 # C01? No, C00 is row 0, col 0.
    # Wait, C is 2x2.
    # C00 = M1 - M4 + M5
    # C01 = M4 + M7
    # C10 = M2 + M3
    # C11 = M1 - M3 + M6
    
    # Let's be precise.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # a_rows[0] corresponds to A00, A01? No.
    # U has rows for all A elements.
    # A1 is 2x2. A1[0,0] is a_rows[0], A1[0,1] is a_rows[0]+1?
    # A is 2x4.
    # A1 is columns 0,1.
    # A1[0,0] = A[0,0] = index 0.
    # A1[0,1] = A[0,1] = index 1.
    # A1[1,0] = A[1,0] = index 4? No, A is 2x4. Row 1 is index 4,5,6,7.
    # U index = i*m + j.
    # A[0,0] -> 0*4+0 = 0.
    # A[0,1] -> 0*4+1 = 1.
    # A[1,0] -> 1*4+0 = 4.
    # A[1,1] -> 1*4+1 = 5.
    # So for A1, indices are [0, 1, 4, 5].
    # A00=0, A01=1, A10=4, A11=5.
    
    # B1 is rows 0,1, cols 0,1,2.
    # B11 is cols 0,1.
    # B is 4x7. V index = j*p + k.
    # B[0,0] -> 0*7+0 = 0.
    # B[0,1] -> 0*7+1 = 1.
    # B[1,0] -> 1*7+0 = 7.
    # B[1,1] -> 1*7+1 = 8.
    # So for B11, indices are [0, 1, 7, 8].
    # B00=0, B01=1, B10=7, B11=8.
    
    # C1 is cols 0,1,2.
    # C11 is cols 0,1.
    # W index = k*n + i.
    # C[0,0] -> k=0, i=0 -> 0.
    # C[0,1] -> k=1, i=0 -> 2.
    # C[1,0] -> k=0, i=1 -> 1.
    # C[1,1] -> k=1, i=1 -> 3.
    # So for C11, indices are [0, 2, 1, 3].
    # C00=0, C01=2, C10=1, C11=3.
    
    # General mapping for a 2x2 block in A (rows r, cols c):
    # A indices: [r*m+c, r*m+c+1, (r+1)*m+c, (r+1)*m+c+1]
    # A00, A01, A10, A11.
    
    # General mapping for a 2x2 block in B (rows r, cols c):
    # B indices: [r*p+c, r*p+c+1, (r+1)*p+c, (r+1)*p+c+1]
    # B00, B01, B10, B11.
    
    # General mapping for a 2x2 block in C (rows i, cols k):
    # C indices: [k*n+i, (k+1)*n+i, k*n+(i+1), (k+1)*n+(i+1)]
    # C00, C01, C10, C11.
    
    # Let's pass the base indices and compute the 4 indices.
    
    pass

# This is getting complicated. Let's just hardcode the 25 columns.

def construct():
    n, m, p = 2, 4, 7
    rank = 25
    U = [[0] * rank for _ in range(n * m)] # 8 x 25
    V = [[0] * rank for _ in range(m * p)] # 28 x 25
    W = [[0] * rank for _ in range(p * n)] # 14 x 25
    
    # Helper to get indices
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i
    
    # Block 1: A1 (rows 0-1, cols 0-1) * B11 (rows 0-1, cols 0-1)
    # Strassen 2x2x2
    # A1 indices:
    A00, A01, A10, A11 = a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1)
    # B11 indices:
    B00, B01, B10, B11 = b_idx(0,0), b_idx(0,1), b_idx(1,0), b_idx(1,1)
    # C11 indices (cols 0-1, rows 0-1):
    C00, C01, C10, C11 = c_idx(0,0), c_idx(1,0), c_idx(0,1), c_idx(1,1)
    
    col = 0
    # M1 = (A00+A11)(B00+B11)
    U[A00][col] += 1; U[A11][col] += 1
    V[B00][col] += 1; V[B11][col] += 1
    W[C00][col] += 1; W[C11][col] += 1
    col += 1
    # M2 = (A10+A11)B00
    U[A10][col] += 1; U[A11][col] += 1
    V[B00][col] += 1
    W[C10][col] += 1
    col += 1
    # M3 = A00(B10-B11)
    U[A00][col] += 1
    V[B10][col] += 1; V[B11][col] -= 1
    W[C10][col] += 1
    col += 1
    # M4 = A11(B01-B10)
    U[A11][col] += 1
    V[B01][col] += 1; V[B10][col] -= 1
    W[C00][col] -= 1; W[C01][col] += 1
    col += 1
    # M5 = (A00+A10)B11
    U[A00][col] += 1; U[A10][col] += 1
    V[B11][col] += 1
    W[C00][col] += 1
    col += 1
    # M6 = (A01-A10)(B00+B10)
    U[A01][col] += 1; U[A10][col] -= 1
    V[B00][col] += 1; V[B10][col] += 1
    W[C11][col] += 1
    col += 1
    # M7 = (A01-A11)(B01+B11)
    U[A01][col] += 1; U[A11][col] -= 1
    V[B01][col] += 1; V[B11][col] += 1
    W[C01][col] += 1
    col += 1
    
    # Block 2: A1 * B12 (rows 0-1, col 2)
    # Schoolbook 2x2x1
    # B12 indices:
    B02, B12 = b_idx(0,2), b_idx(1,2)
    # C12 indices (col 2, rows 0-1):
    C02, C12 = c_idx(2,0), c_idx(2,1)
    
    # M1 = A00*B02
    U[A00][col] += 1
    V[B02][col] += 1
    W[C02][col] += 1
    col += 1
    # M2 = A01*B12
    U[A01][col] += 1
    V[B12][col] += 1
    W[C02][col] += 1
    col += 1
    # M3 = A10*B02
    U[A10][col] += 1
    V[B02][col] += 1
    W[C12][col] += 1
    col += 1
    # M4 = A11*B12
    U[A11][col] += 1
    V[B12][col] += 1
    W[C12][col] += 1
    col += 1
    
    # Block 3: A2 (rows 0-1, cols 2-3) * B21 (rows 2-3, cols 3-4)
    # Strassen 2x2x2
    A00, A01, A10, A11 = a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3)
    B00, B01, B10, B11 = b_idx(2,3), b_idx(2,4), b_idx(3,3), b_idx(3,4)
    C00, C01, C10, C11 = c_idx(3,0), c_idx(4,0), c_idx(3,1), c_idx(4,1)
    
    # M1 = (A00+A11)(B00+B11)
    U[A00][col] += 1; U[A11][col] += 1
    V[B00][col] += 1; V[B11][col] += 1
    W[C00][col] += 1; W[C11][col] += 1
    col += 1
    # M2 = (A10+A11)B00
    U[A10][col] += 1; U[A11][col] += 1
    V[B00][col] += 1
    W[C10][col] += 1
    col += 1
    # M3 = A00(B10-B11)
    U[A00][col] += 1
    V[B10][col] += 1; V[B11][col] -= 1
    W[C10][col] += 1
    col += 1
    # M4 = A11(B01-B10)
    U[A11][col] += 1
    V[B01][col] += 1; V[B10][col] -= 1
    W[C00][col] -= 1; W[C01][col] += 1
    col += 1
    # M5 = (A00+A10)B11
    U[A00][col] += 1; U[A10][col] += 1
    V[B11][col] += 1
    W[C00][col] += 1
    col += 1
    # M6 = (A01-A10)(B00+B10)
    U[A01][col] += 1; U[A10][col] -= 1
    V[B00][col] += 1; V[B10][col] += 1
    W[C11][col] += 1
    col += 1
    # M7 = (A01-A11)(B01+B11)
    U[A01][col] += 1; U[A11][col] -= 1
    V[B01][col] += 1; V[B11][col] += 1
    W[C01][col] += 1
    col += 1
    
    # Block 4: A2 * B22 (rows 2-3, cols 5-6)
    # Strassen 2x2x2
    A00, A01, A10, A11 = a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3)
    B00, B01, B10, B11 = b_idx(2,5), b_idx(2,6), b_idx(3,5), b_idx(3,6)
    C00, C01, C10, C11 = c_idx(5,0), c_idx(6,0), c_idx(5,1), c_idx(6,1)
    
    # M1 = (A00+A11)(B00+B11)
    U[A00][col] += 1; U[A11][col] += 1
    V[B00][col] += 1; V[B11][col] += 1
    W[C00][col] += 1; W[C11][col] += 1
    col += 1
    # M2 = (A10+A11)B00
    U[A10][col] += 1; U[A11][col] += 1
    V[B00][col] += 1
    W[C10][col] += 1
    col += 1
    # M3 = A00(B10-B11)
    U[A00][col] += 1
    V[B10][col] += 1; V[B11][col] -= 1
    W[C10][col] += 1
    col += 1
    # M4 = A11(B01-B10)
    U[A11][col] += 1
    V[B01][col] += 1; V[B10][col] -= 1
    W[C00][col] -= 1; W[C01][col] += 1
    col += 1
    # M5 = (A00+A10)B11
    U[A00][col] += 1; U[A10][col] += 1
    V[B11][col] += 1
    W[C00][col] += 1
    col += 1
    # M6 = (A01-A10)(B00+B10)
    U[A01][col] += 1; U[A10][col] -= 1
    V[B00][col] += 1; V[B10][col] += 1
    W[C11][col] += 1
    col += 1
    # M7 = (A01-A11)(B01+B11)
    U[A01][col] += 1; U[A11][col] -= 1
    V[B01][col] += 1; V[B11][col] += 1
    W[C01][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# (No change to construct signature)
# EVOLVE-BLOCK-END
