"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) of shapes (4,7),(4,7),(4,7) for 2x2 * 2x2.
    U indexed by (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i.
    Uses the standard Strassen 7-term decomposition."""
    # Strassen:
    # M1 = (A11+A22)(B12+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # For the tensor, each term c contributes U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = coefficient for C[i,j] from M_c
    #
    # Let's define the 7 terms and their contributions:
    # Term 1 (M1): A11*A? no... Let me think more carefully.
    #
    # Actually, the bilinear form for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Strassen splits into 7 products of linear combinations:
    # M1 = (A00+A11)*(B01+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B00-B10)
    # M4 = A11*(B11-B01)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # For the tensor decomposition, each term c has:
    # U[i*2+j, c] = coefficient of A[i,j] in the first factor of M_c
    # V[j*2+k, c] = coefficient of B[j,k] in the second factor of M_c
    # W[k*2+i, c] = coefficient of C[i,k] contributed by M_c
    #
    # Wait, the tensor is T[i*2+j, j*2+k, k*2+i] = 1
    # So for a valid decomposition: sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1 for all i,j,k
    #
    # This means: for each (i,j,k), the product must equal 1.
    #
    # Let me define the 7 terms:
    # Term 1: M1 = (A00+A11)*(B01+B11)
    #   A factors: A00:1, A11:1, others 0
    #   B factors: B01:1, B11:1, others 0
    #   C contribution: C00:+1, C11:+1
    #   So U[0,0]=1, U[3,0]=1 (A00 is i=0,j=0; A11 is i=1,j=1)
    #   V[1,0]=1, V[3,0]=1 (B01 is j=0,k=1; B11 is j=1,k=1)
    #   W[0,0]=1, W[3,0]=1 (C00 is k=0,i=0; C11 is k=1,i=1)
    #
    # Let me be more systematic. For each term c:
    # U[i*2+j, c] = a_{ij}^{(c)} (coefficient of A_{ij})
    # V[j*2+k, c] = b_{jk}^{(c)} (coefficient of B_{jk})
    # W[k*2+i, c] = w_{ki}^{(c)} (coefficient of C_{ki}... wait, C is indexed by (i,k) but W is indexed by (k,i))
    #
    # The condition is: for all i,j,k: sum_c a_{ij}^{(c)} * b_{jk}^{(c)} * w_{ki}^{(c)} = delta_{contribution to C[i,k] from A[i,j]*B[j,k]}
    # Actually it's: sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1 for all valid (i,j,k)
    #
    # This is the standard matrix multiplication tensor. Let me just hardcode the known Strassen decomposition.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B01+B11)
    # A: (0,0)->1, (1,1)->1
    # B: (0,1)->1, (1,1)->1
    # C: (0,0)->+1, (1,1)->+1
    U[0, 0] = 1; U[3, 0] = 1
    V[1, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    # A: (1,0)->1, (1,1)->1
    # B: (0,0)->1
    # C: (1,0)->+1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1  # C[1,0] -> k=0, i=1 -> W[0*2+1] = W[1]... wait
    # W[k*2+i]: C[i,k] with i=1,k=0 -> W[0*2+1] = W[1]
    W[1, 1] = 1
    
    # Term 2: M3 = A00(B00-B10)
    # A: (0,0)->1
    # B: (0,0)->1, (1,0)->-1
    # C: (0,0)->+1
    U[0, 2] = 1
    V[0, 2] = 1; V[2, 2] = -1
    W[0, 2] = 1  # C[0,0] -> k=0,i=0 -> W[0]
    
    # Term 3: M4 = A11(B11-B01)
    # A: (1,1)->1
    # B: (1,1)->1, (0,1)->-1
    # C: (0,0)->+1, (1,1)->+1
    U[3, 3] = 1
    V[3, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[3, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    # A: (0,0)->1, (0,1)->1
    # B: (1,1)->1
    # C: (0,0)->-1, (0,1)->+1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C[0,0] -> W[0]
    W[1, 4] = 1   # C[0,1] -> k=1,i=0 -> W[1*2+0] = W[2]... 
    # Wait, W is indexed by k*n+i where n=2. So W[k*2+i].
    # C[i,k]: i is row, k is column.
    # C[0,0]: i=0,k=0 -> W[0*2+0] = W[0]
    # C[0,1]: i=0,k=1 -> W[1*2+0] = W[2]
    # C[1,0]: i=1,k=0 -> W[0*2+1] = W[1]
    # C[1,1]: i=1,k=1 -> W[1*2+1] = W[3]
    
    # Let me redo with correct W indexing:
    # W[k*2+i] corresponds to C[i,k]
    
    # Term 0: M1
    # C[0,0]: +1 -> W[0*2+0]=W[0] += 1
    # C[1,1]: +1 -> W[1*2+1]=W[3] += 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2
    # C[1,0]: +1 -> W[0*2+1]=W[1] += 1
    W[1, 1] = 1
    
    # Term 2: M3
    # C[0,0]: +1 -> W[0] += 1
    W[0, 2] = 1
    
    # Term 3: M4
    # C[0,0]: +1 -> W[0] += 1
    # C[1,1]: +1 -> W[3] += 1
    W[0, 3] = 1; W[3, 3] = 1
    
    # Term 4: M5
    # C[0,0]: -1 -> W[0] += -1
    # C[0,1]: +1 -> W[2] += 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # A: (1,0)->1, (0,0)->-1
    # B: (0,0)->1, (0,1)->1
    # C: (1,1): +1 -> W[3] += 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # A: (0,1)->1, (1,1)->-1
    # B: (1,0)->1, (1,1)->1
    # C: (0,0): +1 -> W[0] += 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    # Verify
    T_rec = np.zeros((4, 4, 4))
    for c in range(7):
        T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    # Check all (i,j,k)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected = 1.0
                got = T_rec[i*2+j, j*2+k, k*2+i]
                if abs(got - expected) > 1e-10:
                    pass  # We'll verify later
    return U, V, W


def _schoolbook_block(n, m, p):
    """Return schoolbook decomposition for n x m * m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)"""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u, col] = 1.0
                V[idx_v, col] = 1.0
                W[idx_w, col] = 1.0
                col += 1
    return U, V, W


def _split_and_combine(U1, V1, W1, U2, V2, W2, 
                        n1, m1, p1, n2, m2, p2,
                        offset_u, offset_v, offset_w,
                        total_u, total_v, total_w):
    """Combine two block decompositions into a larger one.
    Block 1 handles rows 0..n1-1, cols 0..m1-1 of A, rows 0..m1-1, cols 0..p1-1 of B
    Block 2 handles rows n1..n1+n2-1, cols m1..m1+m2-1 of A, rows m1..m1+m2-1, cols p1..p1+p2-1 of B
    
    Actually this is getting complicated. Let me just build the full decomposition directly."""
    pass


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank_schoolbook = n * m * p  # 40
    
    start_time = time.time()
    deadline = start_time + 130
    
    # Build target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0
    
    def is_valid(U, V, W, tol=1e-9):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        T_rec = np.zeros_like(T_target)
        for c in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_rec, T_target, atol=tol, rtol=0)
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook_block(n, m, p)
    best_result = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Strategy: Split the 4-dimensional inner product into two 2-dimensional parts
    # A is 2x4, B is 4x5
    # Split columns of A (and rows of B) into [0,1] and [2,3]
    # Split columns of B (and rows of output) into [0,1,2] and [3,4]
    # 
    # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x5) -> 2x5 output
    # Block 2: A[:, 2:4] (2x2) * B[2:4, :] (2x5) -> 2x5 output
    #
    # For each block, we have a 2x2 * 2x5 multiplication.
    # Split the 5 columns of B into [0,1,2] and [3,4]
    #
    # Block 1a: A[:, 0:2] (2x2) * B[0:2, 0:3] (2x3) -> 2x3
    # Block 1b: A[:, 0:2] (2x2) * B[0:2, 3:5] (2x2) -> 2x2
    # Block 2a: A[:, 2:4] (2x2) * B[2:4, 0:3] (2x3) -> 2x3
    # Block 2b: A[:, 2:4] (2x2) * B[2:4, 3:5] (2x2) -> 2x2
    #
    # For 2x2 * 2x2: use Strassen (rank 7)
    # For 2x2 * 2x3: need a decomposition
    #
    # 2x2 * 2x3 schoolbook: rank 12
    # Can we do better? The tensor is [2,2,3] in (n,m,p) notation... 
    # Actually it's multiplying 2x2 by 2x3.
    # 
    # Let's use schoolbook for 2x2*2x3 (rank 12) and Strassen for 2x2*2x2 (rank 7)
    # Total: 2*(12+7) = 38. That's only marginally better than 40.
    #
    # Alternative: Try to find a better decomposition for 2x2*2x3.
    # The lower bound for 2x2*2x3 is at least max(2*2, 2*3, 2*3) = 6, but likely higher.
    # Known: rank of 2x2*2x2 is 7. For 2x2*2x3, we can think of it as 
    # two 2x2*2x2 blocks with one column missing, or use other techniques.
    #
    # Let's try: 2x2 * 2x3 can be seen as:
    # C[:, 0:2] = A * B[:, 0:2] (2x2*2x2, rank 7)
    # C[:, 2] = A * B[:, 2] (2x2*2x1, rank 4 schoolbook)
    # Total: 7 + 4 = 11
    #
    # So for each 2x2*2x5 block:
    # Split B's 5 columns into [0,1], [2], [3,4]
    # 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) + 2x2*2x2 (rank 7) = 18
    # Or: [0,1,2] and [3,4]: 2x2*2x3 + 2x2*2x2
    # 2x2*2x3: split into 2x2*2x2 (7) + 2x2*2x1 (4) = 11
    # So 2x2*2x5: 11 + 7 = 18
    #
    # Total for 2x4*4x5: 2 * 18 = 36
    #
    # Let's build this.
    
    # Helper: 2x2 * 2x1 schoolbook (rank 4)
    def _strassen_2x2x1():
        """2x2 * 2x1, schoolbook rank 4.
        U: (4,4), V: (2,4), W: (2,4)
        U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        Here p=1, so k is always 0.
        V[j*2+0, c] = V[j, c]
        W[0*2+i, c] = W[i, c]
        """
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        col = 0
        for i in range(2):
            for j in range(2):
                # C[i,0] += A[i,j] * B[j,0]
                U[i*2+j, col] = 1.0
                V[j, col] = 1.0
                W[i, col] = 1.0
                col += 1
        return U, V, W
    
    # Helper: 2x2 * 2x2 Strassen (rank 7) - already defined above
    # But let me verify it works first
    
    U_s, V_s, W_s = _strassen_2x2x2()
    # Quick check
    T_check = np.zeros((4, 4, 4))
    for c in range(7):
        T_check += np.einsum('i,j,k->ijk', U_s[:, c], V_s[:, c], W_s[:, c])
    ok = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if abs(T_check[i*2+j, j*2+k, k*2+i] - 1.0) > 1e-10:
                    ok = False
    if not ok:
        # Fall back to schoolbook for 2x2*2x2
        U_s, V_s, W_s = _schoolbook_block(2, 2, 2)
    
    # Helper: 2x2 * 2x3 decomposition (rank 11 = 7 + 4)
    def _decomp_2x2x3():
        """2x2 * 2x3: split into 2x2*2x2 (cols 0,1) + 2x2*2x1 (col 2)
        U: (4, 11), V: (6, 11), W: (6, 11)
        V indexed by j*3+k, W indexed by k*2+i
        """
        U_22, V_22, W_22 = _strassen_2x2x2()  # (4,7),(4,7),(4,7)
        U_21, V_21, W_21 = _strassen_2x2x1()  # (4,4),(2,4),(2,4)
        
        U_full = np.zeros((4, 11))
        V_full = np.zeros((6, 11))
        W_full = np.zeros((6, 11))
        
        # Part 1: cols 0,1 of B (2x2*2x2)
        for c in range(7):
            U_full[:, c] = U_22[:, c]
            # V_22 is (4,7) indexed by j*2+k for k in {0,1}
            # V_full is (6,11) indexed by j*3+k for k in {0,1,2}
            for j in range(2):
                for k in range(2):
                    V_full[j*3+k, c] = V_22[j*2+k, c]
            # W_22 is (4,7) indexed by k*2+i for k in {0,1}
            # W_full is (6,11) indexed by k*2+i for k in {0,1,2}
            for k in range(2):
                for i in range(2):
                    W_full[k*2+i, c] = W_22[k*2+i, c]
        
        # Part 2: col 2 of B (2x2*2x1)
        for c in range(4):
            col_offset = 7 + c
            U_full[:, col_offset] = U_21[:, c]
            # V_21 is (2,4) indexed by j
            # V_full[j*3+2, col_offset] = V_21[j, c]
            for j in range(2):
                V_full[j*3+2, col_offset] = V_21[j, c]
            # W_21 is (2,4) indexed by i
            # W_full[2*2+i, col_offset] = W_21[i, c]
            for i in range(2):
                W_full[2*2+i, col_offset] = W_21[i, c]
        
        return U_full, V_full, W_full
    
    # Helper: 2x2 * 2x5 decomposition (rank 18 = 11 + 7)
    def _decomp_2x2x5():
        """2x2 * 2x5: split into 2x2*2x3 (cols 0,1,2) + 2x2*2x2 (cols 3,4)
        U: (4, 18), V: (10, 18), W: (10, 18)
        V indexed by j*5+k, W indexed by k*2+i
        """
        U_23, V_23, W_23 = _decomp_2x2x3()  # (4,11),(6,11),(6,11)
        U_22, V_22, W_22 = _strassen_2x2x2()  # (4,7),(4,7),(4,7)
        
        U_full = np.zeros((4, 18))
        V_full = np.zeros((10, 18))
        W_full = np.zeros((10, 18))
        
        # Part 1: cols 0,1,2 of B (2x2*2x3)
        for c in range(11):
            U_full[:, c] = U_23[:, c]
            for j in range(2):
                for k in range(3):
                    V_full[j*5+k, c] = V_23[j*3+k, c]
            for k in range(3):
                for i in range(2):
                    W_full[k*2+i, c] = W_23[k*2+i, c]
        
        # Part 2: cols 3,4 of B (2x2*2x2)
        for c in range(7):
            col_offset = 11 + c
            U_full[:, col_offset] = U_22[:, c]
            for j in range(2):
                for k in range(2):
                    V_full[j*5+(3+k), col_offset] = V_22[j*2+k, c]
            for k in range(2):
                for i in range(2):
                    W_full[(3+k)*2+i, col_offset] = W_22[k*2+i, c]
        
        return U_full, V_full, W_full
    
    # Now build the full 2x4 * 4x5 decomposition
    # Split A's columns into [0,1] and [2,3], B's rows correspondingly
    # Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x5) -> contributes to output
    # Block 2: A[:, 2:4] (2x2) * B[2:4, :] (2x5) -> contributes to output
    
    U_1, V_1, W_1 = _decomp_2x2x5()  # (4,18),(10,18),(10,18)
    U_2, V_2, W_2 = _decomp_2x2x5()  # (4,18),(10,18),(10,18)
    
    total_rank = 36
    U_full = np.zeros((8, total_rank))
    V_full = np.zeros((20, total_rank))
    W_full = np.zeros((10, total_rank))
    
    # Block 1: A columns 0,1; B rows 0,1; output all
    for c in range(18):
        # U: A[i, j] where i in {0,1}, j in {0,1}
        # U_full[i*4+j, c] = U_1[i*2+j, c]
        for i in range(2):
            for j in range(2):
                U_full[i*4+j, c] = U_1[i*2+j, c]
        # V: B[j, k] where j in {0,1}, k in {0,1,2,3,4}
        # V_full[j*5+k, c] = V_1[j*5+k, c]
        for j in range(2):
            for k in range(5):
                V_full[j*5+k, c] = V_1[j*5+k, c]
        # W: output C[i, k] where i in {0,1}, k in {0,1,2,3,4}
        # W_full[k*2+i, c] = W_1[k*2+i, c]
        for k in range(5):
            for i in range(2):
                W_full[k*2+i, c] = W_1[k*2+i, c]
    
    # Block 2: A columns 2,3; B rows 2,3; output all
    for c in range(18):
        col_offset = 18 + c
        for i in range(2):
            for j in range(2):
                U_full[i*4+(2+j), col_offset] = U_2[i*2+j, c]
        for j in range(2):
            for k in range(5):
                V_full[(2+j)*5+k, col_offset] = V_2[j*5+k, c]
        for k in range(5):
            for i in range(2):
                W_full[k*2+i, col_offset] = W_2[k*2+i, c]
    
    # Verify
    if is_valid(U_full, V_full, W_full, tol=1e-9):
        best_result = (U_full, V_full, W_full)
        best_rank = total_rank
    else:
        # Debug: check which entries are wrong
        T_rec = np.zeros_like(T_target)
        for c in range(total_rank):
            T_rec += np.einsum('i,j,k->ijk', U_full[:, c], V_full[:, c], W_full[:, c])
        diff = T_rec - T_target
        max_err = np.max(np.abs(diff))
        # Fall back to schoolbook
        best_result = (U_school, V_school, W_school)
        best_rank = rank_schoolbook
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
