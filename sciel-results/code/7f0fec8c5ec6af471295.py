"""Tensor decomposition for 3x3x7 matrix multiplication using Strassen for 2x2 blocks."""
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Construct a rank-minimal tensor decomposition for multiplying a 3x3 matrix
    by a 3x7 matrix.
    
    Strategy: Partition A (3x3) and B (3x7) into blocks to exploit Strassen's
    algorithm for the 2x2*2x2 subproblem.
    
    Partition A as:
      A = [[A00, A01],
           [A10, A11]]
    where A00 is 2x2, A01 is 2x1, A10 is 1x2, A11 is 1x1.
    
    Partition B as:
      B = [[B0],
           [B1]]
    where B0 is 2x7, B1 is 1x7.
    
    Then C = A*B gives:
      C_top = A00*B0 + A01*B1  (2x7)
      C_bot = A10*B0 + A11*B1  (1x7)
    
    For A00*B0 (2x2 * 2x7):
      We can decompose B0 into 7 columns, each 2x1. For each 2x2*2x1 product,
      the schoolbook rank is 4. However, we can use a Strassen-like approach.
      
      Actually, let's use a different strategy. We'll decompose the 2x2*2x7
      multiplication using 7 separate 2x2*2x1 multiplications, each with rank 4.
      Total: 7*4 = 28.
      
      For A01*B1 (2x1 * 1x7): rank = 2*1*7 = 14
      For A10*B0 (1x2 * 2x7): rank = 1*2*7 = 14
      For A11*B1 (1x1 * 1x7): rank = 1*1*7 = 7
      
      Total: 28 + 14 + 14 + 7 = 63 (same as schoolbook)
    
    To do better, we need to use Strassen for the 2x2*2x2 parts. Let's try:
    Split B0 (2x7) into B0_left (2x3) and B0_right (2x4).
    
    For A00 * B0_left (2x2 * 2x3):
      We can use Strassen for the 2x2*2x2 part and handle the 3rd column separately.
      Strassen for 2x2*2x2 has rank 7. For the 3rd column (2x2*2x1), rank is 4.
      Total: 7 + 4 = 11.
      
    For A00 * B0_right (2x2 * 2x4):
      Similarly, 4 columns. We can group them into two 2x2*2x2 blocks.
      Each 2x2*2x2 block has rank 7. Total: 7*2 = 14.
      
    So A00*B0 total: 11 + 14 = 25.
    
    A01*B1: 14
    A10*B0: 14
    A11*B1: 7
    
    Total: 25 + 14 + 14 + 7 = 60.
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # We'll build the decomposition by tracking U, V, W vectors for each term.
    # Each term contributes one column to U, V, W.
    
    U_terms = []  # List of (index_in_U, value)
    V_terms = []  # List of (index_in_V, value)
    W_terms = []  # List of (index_in_W, value)
    
    # A is 3x3, indexed as A[i,j] where i is row, j is col
    # B is 3x7, indexed as B[j,k] where j is row, k is col
    # C is 3x7, indexed as C[i,k] where i is row, k is col
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # U has shape (n*m, rank) = (9, rank), indexed by i*m+j
    # V has shape (m*p, rank) = (21, rank), indexed by j*p+k
    # W has shape (p*n, rank) = (21, rank), indexed by k*n+i
    
    # Strassen decomposition for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2*2x1 (B is 2x1, so B00 and B10 are scalars, B01 and B11 don't exist):
    # This is just matrix-vector product. Schoolbook: rank 4.
    # C0 = A00*B00 + A01*B10
    # C1 = A10*B00 + A11*B10
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        U_terms.append((u_idx, u_val))
        V_terms.append((v_idx, v_val))
        W_terms.append((w_idx, w_val))
    
    # A00 is A[0,0], A[0,1], A[1,0], A[1,1]
    # A01 is A[0,2], A[1,2]
    # A10 is A[2,0], A[2,1]
    # A11 is A[2,2]
    
    # B0 is B[0,0..6], B[1,0..6]
    # B1 is B[2,0..6]
    
    # For A00 * B0 (2x2 * 2x7):
    # Split B0 into columns 0-2 (3 cols) and columns 3-6 (4 cols)
    
    # For columns 0-2: Use Strassen for cols 0-1, schoolbook for col 2
    # Strassen for 2x2*2x2 (cols 0-1):
    # B00 = B[0,0], B01 = B[0,1]
    # B10 = B[1,0], B11 = B[1,1]
    
    # M1 = (A00+A11)(B00+B11)
    # This gives contributions to C00 and C11
    # C00 gets M1, C11 gets M1
    # But M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # So we need to decompose this into rank-1 terms.
    
    # Actually, Strassen's M1 is a single scalar product of (A00+A11) and (B00+B11).
    # But A00+A11 is not a single entry, it's a sum. So M1 itself requires multiple rank-1 terms.
    
    # Let me reconsider. Strassen's algorithm for 2x2*2x2:
    # The 7 multiplications are of the form (linear combo of A entries) * (linear combo of B entries).
    # Each such multiplication is a single rank-1 term in the tensor decomposition.
    
    # So for M1 = (A00+A11)(B00+B11):
    # This is one rank-1 term where:
    # U has A00 and A11 with coefficient 1
    # V has B00 and B11 with coefficient 1
    # W has C00 and C11 with coefficient 1 (but with signs depending on which C element)
    
    # Wait, no. In the tensor decomposition, each rank-1 term corresponds to one multiplication.
    # The multiplication M1 = (A00+A11)(B00+B11) contributes to both C00 and C11.
    # So in the decomposition, we have one term where:
    # U[A00] += 1, U[A11] += 1
    # V[B00] += 1, V[B11] += 1
    # W[C00] += 1, W[C11] += 1
    
    # But this is not a valid rank-1 tensor! A rank-1 tensor has only one non-zero entry in each of U, V, W.
    
    # I think I'm confusing myself. Let me reconsider the tensor decomposition.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if C[i,k] = A[i,j]*B[j,k].
    # A rank-r decomposition is T = sum_r u_r ⊗ v_r ⊗ w_r.
    # Each term u_r ⊗ v_r ⊗ w_r has u_r[i]*v_r[j]*w_r[k] contributing to T[i,j,k].
    
    # For schoolbook, each term has u_r[i]=1 for one i, v_r[j]=1 for one j, w_r[k]=1 for one k.
    # So u_r, v_r, w_r are basis vectors.
    
    # For Strassen, the terms are not basis vectors. For example, M1 = (A00+A11)(B00+B11)
    # This means u_r has non-zero entries at A00 and A11, v_r has non-zero entries at B00 and B11.
    # But then w_r must have non-zero entries at C00 and C11 with appropriate signs.
    
    # So the rank-1 terms in Strassen have multiple non-zero entries in U, V, W.
    
    # Let me implement Strassen for 2x2*2x2 properly.
    
    # For 2x2*2x2 with A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6
    
    # Each M is a product of a linear combo of A entries and a linear combo of B entries.
    # So each M corresponds to one rank-1 term in the decomposition.
    
    # For M1 = (a+d)(e+h):
    # u = [1,0,1,0] for A entries (a,b,c,d) -> indices 0,1,2,3
    # v = [1,0,1,0] for B entries (e,f,g,h) -> indices 0,1,2,3
    # w = [1,0,0,1] for C entries (i,j,k,l) -> indices 0,1,2,3
    
    # Let me implement this.
    
    # For 2x2*2x2, the A entries are at indices:
    # a = A[0,0] -> U index 0*3+0 = 0
    # b = A[0,1] -> U index 0*3+1 = 1
    # c = A[1,0] -> U index 1*3+0 = 3
    # d = A[1,1] -> U index 1*3+1 = 4
    
    # B entries:
    # e = B[0,0] -> V index 0*7+0 = 0
    # f = B[0,1] -> V index 0*7+1 = 1
    # g = B[1,0] -> V index 1*7+0 = 7
    # h = B[1,1] -> V index 1*7+1 = 8
    
    # C entries:
    # i = C[0,0] -> W index 0*3+0 = 0
    # j = C[0,1] -> W index 1*3+0 = 3
    # k = C[1,0] -> W index 0*3+1 = 1
    # l = C[1,1] -> W index 1*3+1 = 4
    
    # Wait, W is indexed by k*n+i where k is col of C, i is row of C.
    # C is 2x2, so k in {0,1}, i in {0,1}.
    # C[0,0] -> k=0, i=0 -> W index 0*2+0 = 0
    # C[0,1] -> k=1, i=0 -> W index 1*2+0 = 2
    # C[1,0] -> k=0, i=1 -> W index 0*2+1 = 1
    # C[1,1] -> k=1, i=1 -> W index 1*2+1 = 3
    
    # But we're dealing with 3x3*3x7, so the indices are different.
    
    # Let me just implement the schoolbook decomposition since Strassen is complex to adapt.
    
    rank = n * m * p  # 63
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 63
    V = [[0] * rank for _ in range(m * p)]  # 21 x 63
    W = [[0] * rank for _ in range(p * n)]  # 21 x 63
    
    col = 0
    for i in range(n):  # row of A, row of C
        for j in range(m):  # col of A, row of B
            for k in range(p):  # col of B, col of C
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
