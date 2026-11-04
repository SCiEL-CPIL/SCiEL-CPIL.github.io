"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns factor matrices of shapes (4, 7), (4, 7), (4, 7) for the 2x2x2 tensor.
    
    The 7 rank-1 terms are:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    For C = AB where A is 2x2, B is 2x2:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # A indexing: A11=0, A12=1, A21=2, A22=3 (row-major, 2x2)
    # B indexing: B11=0, B12=1, B21=2, B22=3 (row-major, 2x2)
    # C indexing: C11=0, C12=1, C21=2, C22=3 (row-major, 2x2)
    
    # W follows trace convention: W[k*n+i, c] where n=2, so W[2*k+i, c]
    # C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # In trace convention for C: output index is (k, i) -> W[k*2 + i, c]
    # Wait, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So for C[i,k], the output index is k*n + i where n=2 (the n from n x m * m x p -> p x n)
    # Actually for 2x2*2x2 -> 2x2: n=2, m=2, p=2
    # C[i,k], output index = k*n + i = 2k + i
    
    U = np.zeros((4, 7))  # n*m = 2*2 = 4
    V = np.zeros((4, 7))  # m*p = 2*2 = 4
    W = np.zeros((4, 7))  # p*n = 2*2 = 4
    
    # A indices: i*m + j where i,j in {0,1}, m=2
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: j*p + k where j,k in {0,1}, p=2
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C output: k*n + i where n=2
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A11+A22)(B11+B22)
    # A11=A[0,0]=0, A22=A[1,1]=3
    # B11=B[0,0]=0, B22=B[1,1]=3
    # C11 += M1, C22 += M1
    # C11 index: k=0,i=0 -> 0; C22 index: k=1,i=1 -> 3
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (A21+A22)B11
    # A21=A[1,0]=2, A22=A[1,1]=3
    # B11=B[0,0]=0
    # C21 += M2, C22 -= M2
    # C21 index: k=1,i=0 -> 2; C22 index: k=1,i=1 -> 3
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1               # B11
    W[2, 1] = 1; W[3, 1] = -1 # C21, -C22
    
    # M3 = A11(B21-B22)
    # A11=A[0,0]=0
    # B21=B[1,0]=2, B22=B[1,1]=3
    # C12 += M3, C22 += M3
    # C12 index: k=1,i=0 -> 2; C22 index: k=1,i=1 -> 3
    U[0, 2] = 1               # A11
    V[2, 2] = 1; V[3, 2] = -1 # B21, -B22
    W[2, 2] = 1; W[3, 2] = 1  # C21? Wait...
    
    # Let me redo this more carefully.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me verify:
    # M1 = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # M4 = A22*B12 - A22*B11
    # M5 = A11*B22 + A12*B22
    # M7 = A12*B21 + A12*B22 - A22*B21 - A22*B22
    
    # C11 = M1 + M4 - M5 + M7
    # = (A11*B11 + A11*B22 + A22*B11 + A22*B22)
    # + (A22*B12 - A22*B11)
    # - (A11*B22 + A12*B22)
    # + (A12*B21 + A12*B22 - A22*B21 - A22*B22)
    # = A11*B11 + A11*B22 + A22*B11 + A22*B22 + A22*B12 - A22*B11 - A11*B22 - A12*B22 + A12*B21 + A12*B22 - A22*B21 - A22*B22
    # = A11*B11 + A12*B21
    # = C11 ✓
    
    # C12 = M3 + M5
    # = A11*B21 - A11*B22 + A11*B22 + A12*B22
    # = A11*B21 + A12*B22
    # Hmm, that's not C12 = A11*B12 + A12*B22
    
    # I think I have the Strassen formulas slightly off. Let me look up the standard ones.
    # Standard Strassen for C = AB, A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C = [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    # So: C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # Let me verify C12:
    # p3 + p5 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh
    # C12 should be a*f + b*h = af + bh ✓
    
    # Great, so the formulas are correct. Let me redo with a=A11, b=A12, c=A21, d=A22
    # e=B11, f=B12, g=B21, h=B22
    
    # A indices: a=0, b=1, c=2, d=3
    # B indices: e=0, f=1, g=2, h=3
    # C output indices (k*n+i, n=2):
    # C11: k=0,i=0 -> 0
    # C12: k=1,i=0 -> 2
    # C21: k=0,i=1 -> 1
    # C22: k=1,i=1 -> 3
    
    # Wait, I need to be more careful about the output indexing.
    # The task says W[k*n+i, r] where the output is C[i,k] (i-th row, k-th column of C)
    # For 2x2 * 2x2 -> 2x2: n=2 (rows of first matrix = columns of output), p=2 (cols of second = rows of output)
    # C has shape (n, p) = (2, 2), C[i, k]
    # Output index in W: k*n + i
    
    # C11 = C[0,0]: k=0, i=0 -> index 0
    # C12 = C[0,1]: k=1, i=0 -> index 2
    # C21 = C[1,0]: k=0, i=1 -> index 1
    # C22 = C[1,1]: k=1, i=1 -> index 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a+d)(e+h)
    # A: a=0, d=3; B: e=0, h=3
    # C11 += p1, C22 += p1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (c+d)e
    # A: c=2, d=3; B: e=0
    # C21 += p2, C22 -= p2
    # C21 index: k=0,i=1 -> 1; C22 index: k=1,i=1 -> 3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # p3 = a(f-h)
    # A: a=0; B: f=1, h=3
    # C12 += p3, C22 += p3
    # C12 index: k=1,i=0 -> 2; C22 index: k=1,i=1 -> 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # p4 = d(g-e)
    # A: d=3; B: g=2, e=0
    # C11 += p4, C21 += p4
    # C11 index: 0; C21 index: 1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # p5 = (a+b)h
    # A: a=0, b=1; B: h=3
    # C11 -= p5, C12 += p5
    # C11 index: 0; C12 index: 2
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # p6 = (c-a)(e+f)
    # A: c=2, a=0; B: e=0, f=1
    # C22 += p6
    # C22 index: 3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # p7 = (b-d)(g+h)
    # A: b=1, d=3; B: g=2, h=3
    # C11 += p7
    # C11 index: 0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1]
    # Compute the tensor from the decomposition
    T_decomp = np.zeros((n, p, n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    # Contribution: u[i*m+j] * v[j*p+k] * w[k*n+i]
                    T_decomp[i, k, j, 0, 0] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    # The target tensor for matrix multiplication:
    # T[i, k, j] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # Actually the tensor is T[i,k,j] = 1 for all valid (i,k,j)
    # Since C[i,k] = sum_j A[i,j] B[j,k], each (i,j,k) triple contributes
    T_target = np.zeros((n, p, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, k, j] = 1
    
    # Check: sum over r of u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] should equal 1
    # for all (i,j,k)
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    return error < 1e-9

def _block_decomp(n, m, p):
    """
    Build a block-based decomposition.
    
    Split the 4x4x7 problem into blocks and use Strassen where beneficial.
    """
    n, m, p = 4, 4, 7
    
    # Strategy: Split into 2x2 blocks for the first two dimensions
    # A is 4x4, B is 4x7
    # Split A into 2x2 blocks of size 2x2
    # Split B into blocks of size 2x7 and 2x7
    
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B1], [B2]] where B1 is 2x7, B2 is 2x7
    
    # C = A*B
    # C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    # C is 4x7
    
    # For the 2x2 * 2x7 part, we can split B1 and B2 into 2x3 and 2x4 blocks
    # Or we can handle 2x2 * 2x7 directly.
    
    # Let's use a simpler approach: split into smaller blocks and use Strassen
    # for the 2x2x2 sub-problems, then combine.
    
    # Alternative: Use the known result that R(4,4,7) can be done with rank < 112.
    # The schoolbook rank is 4*4*7 = 112.
    
    # Let me try a different splitting strategy.
    # Split m=4 into 2+2, and p=7 into 3+4.
    
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # B is 4x7, split rows into 2+2 and columns into 3+4:
    # B11 (2x3), B12 (2x4), B21 (2x3), B22 (2x4)
    
    # C is 4x7
    # C11 = A11*B11 + A12*B21  (2x3)
    # C12 = A11*B12 + A12*B22  (2x4)
    # C21 = A21*B11 + A22*B21  (2x3)
    # C22 = A21*B12 + A22*B22  (2x4)
    
    # Each 2x2 * 2x3 product has rank 2*2*3 = 12 in schoolbook.
    # Each 2x2 * 2x4 product has rank 2*2*4 = 16 in schoolbook.
    # Total: 2*12 + 2*16 = 56. Plus the cross terms...
    
    # Actually, let me just try to build a decomposition using Strassen for 2x2x2
    # and handle the remaining dimensions with schoolbook.
    
    # Let me try a cleaner approach: use the bilinear rank decomposition
    # by viewing the problem as a sum of rank-1 tensors.
    
    # For now, let me just return the schoolbook decomposition as a fallback,
    # but try to improve it.
    
    # Actually, let me try a specific construction:
    # Use Strassen for the 4x4x4 sub-problem and schoolbook for 4x4x3.
    
    # 4x4x4: Use two 2x2 Strassen in each "layer"
    # Actually, 4x4x4 can be done with rank 49 using Strassen recursively,
    # but let me just use a direct construction.
    
    # For 4x4 * 4x7:
    # Split B into B_left (4x4) and B_right (4x3)
    # C = [A*B_left, A*B_right]
    
    # A*B_left: 4x4 * 4x4 -> 4x4, rank 49 via Strassen (approx)
    # A*B_right: 4x4 * 4x3 -> 4x3, rank 48 via schoolbook
    
    # Total rank: 49 + 48 = 97
    
    # Let me build the 4x4x4 Strassen decomposition.
    # 4x4x4 can be decomposed using 2x2x2 Strassen blocks.
    
    # Split 4x4 into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], each 2x2
    # B = [[B11, B12], [B21, B22]], each 2x2
    # C = [[C11, C12], [C21, C22]], each 2x2
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each Cxy is 2x2, computed as sum of two 2x2*2x2 products.
    # Using Strassen for each 2x2*2x2 product gives 7 multiplications each.
    # But we can combine them.
    
    # Actually, for 4x4*4x4, we can use Strassen recursively:
    # R(4,4,4) <= 7 * R(2,2,2) = 7 * 8 = 56? No, that's not right.
    # R(4,4,4) <= 2^3 * R(2,2,2) ... no.
    # Standard: R(2n,2n,2n) <= 7 R(n,n,n) for Strassen.
    # So R(4,4,4) <= 7 * R(2,2,2) = 7 * 8 = 56? 
    # But R(2,2,2) = 7 (Strassen), so R(4,4,4) <= 7*7 = 49.
    
    # Yes! R(4,4,4) <= 49 using Strassen recursively.
    
    # Let me build this.
    
    # For 4x4x4, using recursive Strassen:
    # Split into 2x2 blocks, use 7 Strassen multiplications of 2x2 matrices.
    # Each 2x2 multiplication uses 7 rank-1 terms.
    # Total: 7 * 7 = 49 rank-1 terms.
    
    # But the factors need to be constructed carefully.
    
    # Let me define the 4x4x4 Strassen decomposition.
    # A = [[a11,a12,a13,a14],
    #      [a21,a22,a23,a24],
    #      [a31,a32,a33,a34],
    #      [a41,a42,a43,a44]]
    
    # B = [[b11,b12,b13,b14],
    #      [b21,b22,b23,b24],
    #      [b31,b32,b33,b34],
    #      [b41,b42,b43,b44]]
    
    # C = A*B
    
    # Strassen: split into 2x2 blocks
    # A11 = [[a11,a12],[a21,a22]], A12 = [[a13,a14],[a23,a24]]
    # A21 = [[a31,a32],[a41,a42]], A22 = [[a33,a34],[a43,a44]]
    # Similarly for B.
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    
    # Each Mi is a 2x2 matrix computed via 2x2*2x2 multiplication.
    # Using Strassen for each gives 7 terms per Mi, total 49 terms.
    
    # For the tensor decomposition, each rank-1 term corresponds to one element
    # of the 2x2 result of a Strassen multiplication.
    
    # This is getting complex. Let me just implement it directly.
    
    def _strassen_2x2_factors():
        """
        Return the 7 rank-1 terms for 2x2*2x2 -> 2x2.
        Each term is (u, v, w) where u is 4-vector (A entries), v is 4-vector (B entries),
        w is 4-vector (C output entries with signs).
        A indexing: 0=A11, 1=A12, 2=A21, 3=A22
        B indexing: 0=B11, 1=B12, 2=B21, 3=B22
        C output indexing (k*n+i, n=2): 0=C11, 1=C21, 2=C12, 3=C22
        """
        terms = []
        # p1 = (a+d)(e+h)
        terms.append((
            [1, 0, 0, 1],  # a + d
            [1, 0, 0, 1],  # e + h
            [1, 0, 0, 1]   # C11 + C22 (indices 0 and 3)
        ))
        # p2 = (c+d)e
        terms.append((
            [0, 0, 1, 1],  # c + d
            [1, 0, 0, 0],  # e
            [0, 1, 0, -1]  # C21 - C22 (indices 1 and 3)
        ))
        # p3 = a(f-h)
        terms.append((
            [1, 0, 0, 0],  # a
            [0, 1, 0, -1], # f - h
            [0, 0, 1, 1]   # C12 + C22 (indices 2 and 3)
        ))
        # p4 = d(g-e)
        terms.append((
            [0, 0, 0, 1],  # d
            [0, 0, 1, -1], # g - e
            [1, 1, 0, 0]   # C11 + C21 (indices 0 and 1)
        ))
        # p5 = (a+b)h
        terms.append((
            [1, 1, 0, 0],  # a + b
            [0, 0, 0, 1],  # h
            [-1, 0, 1, 0]  # -C11 + C12 (indices 0 and 2)
        ))
        # p6 = (c-a)(e+f)
        terms.append((
            [0, 0, 1, -1], # c - a
            [1, 1, 0, 0],  # e + f
            [0, 0, 0, 1]   # C22 (index 3)
        ))
        # p7 = (b-d)(g+h)
        terms.append((
            [0, 1, 0, -1], # b - d
            [0, 0, 1, 1],  # g + h
            [1, 0, 0, 0]   # C11 (index 0)
        ))
        return terms
    
    def _build_4x4x4_strassen():
        """Build rank-49 decomposition for 4x4*4x4 -> 4x4."""
        n, m, p = 4, 4, 4
        rank = 49
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # A indexing: i*m + j, i in 0..3, j in 0..3
        # A11: rows 0-1, cols 0-1 -> indices 0,1,2,3
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7 (wait, i*m+j: row 0 col 2 = 0*4+2=2, row 0 col 3 = 3, row 1 col 2 = 6, row 1 col 3 = 7)
        # Hmm, i*m+j with m=4: A[i,j] = index i*4+j
        # A11 (rows 0-1, cols 0-1): indices 0,1,4,5
        # A12 (rows 0-1, cols 2-3): indices 2,3,6,7
        # A21 (rows 2-3, cols 0-1): indices 8,9,12,13
        # A22 (rows 2-3, cols 2-3): indices 10,11,14,15
        
        # B indexing: j*p + k, j in 0..3, k in 0..3
        # B[j,k] = index j*4+k
        # B11 (rows 0-1, cols 0-1): indices 0,1,4,5
        # B12 (rows 0-1, cols 2-3): indices 2,3,6,7
        # B21 (rows 2-3, cols 0-1): indices 8,9,12,13
        # B22 (rows 2-3, cols 2-3): indices 10,11,14,15
        
        # C output: k*n + i, k in 0..3, i in 0..3
        # C[i,k] = index k*4+i
        # C11 (rows 0-1, cols 0-1): C[0,0]=0, C[0,1]=4, C[1,0]=1, C[1,1]=5 -> indices 0,1,4,5
        # C12 (rows 0-1, cols 2-3): C[0,2]=8, C[0,3]=12, C[1,2]=9, C[1,3]=13 -> indices 8,9,12,13
        # C21 (rows 2-3, cols 0-1): C[2,0]=2, C[2,1]=6, C[3,0]=3, C[3,1]=7 -> indices 2,3,6,7
        # C22 (rows 2-3, cols 2-3): C[2,2]=10, C[2,3]=14, C[3,2]=11, C[3,3]=15 -> indices 10,11,14,15
        
        # For each of the 7 Strassen terms at the 4x4 level,
        # we need to compute a 2x2 matrix product, which itself uses 7 terms.
        
        # Let me define the block indices.
        # A blocks (2x2 each, flattened in row-major):
        # A11: [0,1,4,5] (A[0,0],A[0,1],A[1,0],A[1,1])
        # A12: [2,3,6,7]
        # A21: [8,9,12,13]
        # A22: [10,11,14,15]
        
        # B blocks:
        # B11: [0,1,4,5]
        # B12: [2,3,6,7]
        # B21: [8,9,12,13]
        # B22: [10,11,14,15]
        
        # C blocks (output, k*n+i with n=4):
        # C11: [0,1,4,5]
        # C12: [8,9,12,13]
        # C21: [2,3,6,7]
        # C22: [10,11,14,15]
        
        A_blocks = {
            'A11': [0, 1, 4, 5],
            'A12': [2, 3, 6, 7],
            'A21': [8, 9, 12, 13],
            'A22': [10, 11, 14, 15]
        }
        B_blocks = {
            'B11': [0, 1, 4, 5],
            'B12': [2, 3, 6, 7],
            'B21': [8, 9, 12, 13],
            'B22': [10, 11, 14, 15]
        }
        C_blocks = {
            'C11': [0, 1, 4, 5],
            'C12': [8, 9, 12, 13],
            'C21': [2, 3, 6, 7],
            'C22': [10, 11, 14, 15]
        }
        
        # Strassen terms at 4x4 level:
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        # M3 = A11(B21-B22) -> C12 += M3, C22 += M3
        # M4 = A22(B12-B11) -> C11 += M4, C21 += M4
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        
        # Each Mi is a 2x2 matrix = sum of 7 rank-1 terms from 2x2 Strassen.
        # For a 2x2 product X*Y where X and Y are 2x2 matrices (each with 4 elements),
        # the 7 terms are given by _strassen_2x2_factors().
        
        # For each Strassen term at the 4x4 level, we need to:
        # 1. Identify which A-block elements and B-block elements are involved.
        # 2. For each of the 7 sub-terms, add to U, V, W.
        
        # Let me define a helper to add a 2x2x2 Strassen decomposition
        # with specific block mappings.
        
        def add_2x2_term(col_start, a_indices, b_indices, c_indices, sign=1):
            """
            Add 7 rank-1 terms for a 2x2*2x2 -> 2x2 product.
            a_indices: list of 4 A-element indices (for the 2x2 A block)
            b_indices: list of 4 B-element indices (for the 2x2 B block)
            c_indices: list of 4 C-output indices (for the 2x2 C block)
            sign: multiplier for all terms
            Returns the next column index.
            """
            strassen_terms = _strassen_2x2_factors()
            for t in strassen_terms:
                u_vec, v_vec, w_vec = t
                for idx, val in enumerate(u_vec):
                    if val != 0:
                        U[a_indices[idx], col_start + t_idx] = val * sign
                for idx, val in enumerate(v_vec):
                    if val != 0:
                        V[b_indices[idx], col_start + t_idx] = val * sign
                for idx, val in enumerate(w_vec):
                    if val != 0:
                        W[c_indices[idx], col_start + t_idx] = val * sign
                t_idx += 1
            return col_start + 7
        
        # This is getting messy. Let me take a different, cleaner approach.
        # I'll just directly enumerate all 49 terms.
        
        # For each of the 7 Strassen multiplications at the 4x4 level,
        # and for each of the 7 Strassen terms at the 2x2 level,
        # I have one rank-1 term.
        
        # Let me define the mappings more carefully.
        
        # Strassen 4x4 level term t (0-6) uses certain A and B blocks.
        # Within that, Strassen 2x2 level term s (0-6) uses certain elements.
        
        # A-block element mapping:
        # For A11: local index 0->A[0,0], 1->A[0,1], 2->A[1,0], 3->A[1,1]
        # Global indices: 0, 1, 4, 5
        
        # Let me just hardcode all 49 terms. This is tedious but reliable.
        
        # Actually, let me use a programmatic approach.
        
        # For each Strassen-4x4 term, define the A and B block combinations.
        # Then for each Strassen-2x2 term within, define the element combinations.
        
        # Strassen-4x4 terms:
        # Term 0 (M1): A = A11+A22, B = B11+B22, C: C11+=, C22+=
        # Term 1 (M2): A = A21+A22, B = B11, C: C21+=, C22-=
        # Term 2 (M3): A = A11, B = B21-B22, C: C12+=, C22+=
        # Term 3 (M4): A = A22, B = B12-B11, C: C11+=, C21+=
        # Term 4 (M5): A = A11+A12, B = B22, C: C11-=, C12+=
        # Term 5 (M6): A = A21-A11, B = B11+B12, C: C22+=
        # Term 6 (M7): A = A12-A22, B = B21+B22, C: C11+=
        
        # For "A = A11+A22", this means the A-vector for the 2x2 product has
        # elements from both A11 and A22 added together. In the tensor decomposition,
        # this is not a single 2x2 block but a sum of two 2x2 blocks.
        
        # This is the key issue: Strassen requires multiplying sums of blocks,
        # which in the tensor decomposition means we can't just use one 2x2 Strassen
        # decomposition. We need to expand.
        
        # Actually, no. In the tensor decomposition, each rank-1 term is:
        # u ⊗ v ⊗ w where u, v, w are vectors.
        
        # For M1 = (A11+A22)(B11+B22), the "A side" is A11+A22, which is a 2x2 matrix
        # whose entries are sums of entries from A11 and A22.
        
        # In the rank-1 decomposition, for each element of the result 2x2 matrix,
        # we have terms like (sum of A entries) * (sum of B entries).
        
        # This expands to a sum of products of individual A entries and B entries,
        # which is exactly what the 2x2 Strassen decomposition provides, but with
        # the A-vector being the concatenation of A11 and A22 entries (with appropriate signs),
        # and similarly for B.
        
        # Wait, no. The 2x2 Strassen decomposition assumes A and B are 2x2 matrices
        # with 4 entries each. If A = A11 + A22, then A has entries that are sums of
        # pairs of original A entries. In the tensor decomposition, this means:
        
        # Instead of U having a 1 in the position for A11[0,0] and A22[0,0],
        # we need U to have 1 in both positions (since A[0,0] = A11[0,0] + A22[0,0]).
        
        # So for M1, the A-vector for the 2x2 Strassen decomposition is:
        # [A11[0,0] + A22[0,0], A11[0,1] + A22[0,1], A11[1,0] + A22[1,0], A11[1,1] + A22[1,1]]
        
        # In the U matrix, for each of the 7 Strassen-2x2 terms, we put 1 in the
        # positions corresponding to A11 elements and A22 elements.
        
        # This is correct! The U matrix has 16 rows (4x4), and for a term involving
        # A11+A22, we set U[A11_idx] = 1 and U[A22_idx] = 1 for the relevant elements.
        
        # Let me implement this properly.
        
        col = 0
        for s4 in range(7):  # Strassen-4x4 term
            # Determine A-block combination, B-block combination, C-block contributions
            # A_comb: list of (block_name, sign)
            # B_comb: list of (block_name, sign)
            # C_comb: list of (block_name, sign)
            
            if s4 == 0:  # M1 = (A11+A22)(B11+B22)
                A_comb = [('A11', 1), ('A22', 1)]
                B_comb = [('B11', 1), ('B22', 1)]
                C_comb = [('C11', 1), ('C22', 1)]
            elif s4 == 1:  # M2 = (A21+A22)B11
                A_comb = [('A21', 1), ('A22', 1)]
                B_comb = [('B11', 1)]
                C_comb = [('C21', 1), ('C22', -1)]
            elif s4 == 2:  # M3 = A11(B21-B22)
                A_comb = [('A11', 1)]
                B_comb = [('B21', 1), ('B22', -1)]
                C_comb = [('C12', 1), ('C22', 1)]
            elif s4 == 3:  # M4 = A22(B12-B11)
                A_comb = [('A22', 1)]
                B_comb = [('B12', 1), ('B11', -1)]
                C_comb = [('C11', 1), ('C21', 1)]
            elif s4 == 4:  # M5 = (A11+A12)B22
                A_comb = [('A11', 1), ('A12', 1)]
                B_comb = [('B22', 1)]
                C_comb = [('C11', -1), ('C12', 1)]
            elif s4 == 5:  # M6 = (A21-A11)(B11+B12)
                A_comb = [('A21', 1), ('A11', -1)]
                B_comb = [('B11', 1), ('B12', 1)]
                C_comb = [('C22', 1)]
            elif s4 == 6:  # M7 = (A12-A22)(B21+B22)
                A_comb = [('A12', 1), ('A22', -1)]
                B_comb = [('B21', 1), ('B22', 1)]
                C_comb = [('C11', 1)]
            
            # For this Strassen-4x4 term, we compute a 2x2 product.
            # The A-side is a 2x2 matrix formed by the combination in A_comb.
            # The B-side is a 2x2 matrix formed by the combination in B_comb.
            # The result contributes to C blocks as specified in C_comb.
            
            # For each of the 7 Strassen-2x2 terms:
            strassen_2x2 = _strassen_2x2_factors()
            for s2 in range(7):
                u_vec, v_vec, w_vec = strassen_2x2[s2]
                
                # u_vec: 4-element vector for the 2x2 A matrix
                # v_vec: 4-element vector for the 2x2 B matrix
                # w_vec: 4-element vector for the 2x2 C output
                
                # Build U contribution:
                for local_a in range(4):
                    if u_vec[local_a] != 0:
                        for block_name, sign in A_comb:
                            a_idx = A_blocks[block_name][local_a]
                            U[a_idx, col] += u_vec[local_a] * sign
                
                # Build V contribution:
                for local_b in range(4):
                    if v_vec[local_b] != 0:
                        for block_name, sign in B_comb:
                            b_idx = B_blocks[block_name][local_b]
                            V[b_idx, col] += v_vec[local_b] * sign
                
                # Build W contribution:
                for local_c in range(4):
                    if w_vec[local_c] != 0:
                        for block_name, sign in C_comb:
                            c_idx = C_blocks[block_name][local_c]
                            W[c_idx, col] += w_vec[local_c] * sign
                
                col += 1
        
        assert col == 49, f"Expected 49 columns, got {col}"
        return U, V, W
    
    U_4x4x4, V_4x4x4, W_4x4x4 = _build_4x4x4_strassen()
    
    # Now build 4x4x3 schoolbook decomposition.
    n2, m2, p2 = 4, 4, 3
    rank2 = n2 * m2 * p2
    U2 = np.zeros((n2 * m2, rank2))
    V2 = np.zeros((m2 * p2, rank2))
    W2 = np.zeros((p2 * n2, rank2))
    col2 = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j, col2] = 1
                V2[j * p2 + k, col2] = 1
                W2[k * n2 + i, col2] = 1
                col2 += 1
    
    # Combine: 4x4*4x7 = [4x4*4x4, 4x4*4x3]
    # U: same for both parts (A is 4x4 in both)
    # V: 4x4 part uses V_4x4x4 (4x4), 4x3 part uses V2 (4x3)
    #    But V has shape (m*p, rank) = (4*7, rank) = (28, rank)
    #    For the 4x4x4 part, B is 4x4, so V indices are j*p+k with p=4
    #    For the 4x4x3 part, B is 4x3, so V indices are j*3+k
    #    These are different! We need to map them to the full 4x7 B.
    
    # In the full problem, B is 4x7. The 4x4x4 part uses B[:, 0:4] and the
    # 4x4x3 part uses B[:, 4:7].
    
    # So for V, the full B has indices j*7+k for j in 0..3, k in 0..6.
    # For the 4x4x4 part, B entries are at j*7+k for k in 0..3.
    # For the 4x4x3 part, B entries are at j*7+k for k in 4..6.
    
    # Similarly for W:
    # Full C is 4x7, output index k*4+i for k in 0..6, i in 0..3.
    # For the 4x4x4 part, C entries are at k*4+i for k in 0..3.
    # For the 4x4x3 part, C entries are at k*4+i for k in 4..6.
    
    # Let me rebuild V and W with the correct indexing.
    
    rank_total = 49 + 48  # 97
    U_full = np.zeros((16, rank_total))  # n*m = 4*4 = 16
    V_full = np.zeros((28, rank_total))  # m*p = 4*7 = 28
    W_full = np.zeros((28, rank_total))  # p*n = 7*4 = 28
    
    # Copy U from 4x4x4 part
    U_full[:, :49] = U_4x4x4
    # U is the same for the 4x4x3 part
    U_full[:, 49:] = U2
    
    # V for 4x4x4 part: map j*4+k (k in 0..3) to j*7+k
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V_full[j * 7 + k, r] = V_4x4x4[j * 4 + k, r]
    
    # V for 4x4x3 part: map j*3+k (k in 0..2) to j*7+(4+k)
    for j in range(4):
        for k in range(3):
            for r in range(48):
                V_full[j * 7 + (4 + k), 49 + r] = V2[j * 3 + k, r]
    
    # W for 4x4x4 part: map k*4+i (k in 0..3) to k*4+i (same, since p=7, n=4)
    # Wait, W index is k*n + i where n=4. For the 4x4x4 part, k in 0..3.
    # For the full problem, k in 0..6.
    # W_4x4x4 has shape (4*4, 49) = (16, 49), with index k*4+i for k in 0..3.
    # In the full W, this maps to k*4+i for k in 0..3.
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W_full[k * 4 + i, r] = W_4x4x4[k * 4 + i, r]
    
    # W for 4x4x3 part: map k*4+i (k in 0..2) to (4+k)*4+i
    for k in range(3):
        for i in range(4):
            for r in range(48):
                W_full[(4 + k) * 4 + i, 49 + r] = W2[k * 4 + i, r]
    
    return U_full, V_full, W_full

def construct():
    """
    Construct a tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Uses Strassen for the 4x4x4 sub-problem (rank 49) and schoolbook for
    the 4x4x3 sub-problem (rank 48), for a total rank of 97.
    """
    start_time = time.time()
    deadline = 150.0 - 10.0  # Safety margin
    
    try:
        U, V, W = _block_decomp(*DIMENSIONS)
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        U, V, W = _schoolbook_decomp(n, m, p)
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
