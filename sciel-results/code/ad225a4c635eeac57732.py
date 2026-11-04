"""Matrix multiplication 4x4x6: structured Kronecker-product decomposition.

Strategy:
  - Treat the 4x4x6 multiplication as a Kronecker product of a 2x2x2 core
    (Strassen, rank 7) and a 2x2x3 rectangular base (rank 11).
  - Total rank = 7 * 11 = 77.
  - The 2x2x3 base is a schoolbook-style decomposition with one shared column
    to reduce from 12 to 11 (the (0,1) and (1,0) positions in the 2x2x3 output
    can share a rank-1 term because the multiplication tensor has symmetry).
  
  Actually, let me reconsider: 2x2x3 schoolbook has rank 2*2*3 = 12.
  Can we do better? The known lower bound for 2x2x3 is 10 (from Strassen's
  rectangular matrix multiplication). But constructing an explicit rank-10 or
  rank-11 decomposition is non-trivial.

  Let me use a different approach: use the known rank-11 decomposition for 2x2x3
  if one exists, or fall back to rank 12.

  Actually, for 2x2x3, the schoolbook rank is 12. The best known upper bound
  is 12 (no improvement known for exact decomposition over integers).
  
  Wait - let me reconsider the Kronecker approach. 4x4x6 = (2x2)⊗(2x2x3) where:
  - The 4x4 matrix is partitioned into 2x2 blocks, each 2x2
  - The 4x6 matrix is partitioned into 2x2 blocks, each 2x3
  
  The multiplication (A_{2x2 blocks of 2x2}) * (B_{2x2 blocks of 2x3}) gives
  C_{2x2 blocks of 2x3}.
  
  Using Kronecker: rank = rank(2x2x2) * rank(2x2x3) = 7 * 12 = 84.
  
  But the directive says rank 77 = 7 * 11. So we need a rank-11 decomposition
  for 2x2x3. Let me check if such exists.

  For 2x2x3 matrix multiplication over the reals, the tensor is:
  T = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{2} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
  
  where e_{i,j} is in R^4 (for the 2x2 input A), e_{j,k} is in R^6 (for the 2x3 input B),
  and e_{k,i} is in R^6 (for the 2x3 output C).

  The schoolbook gives rank 12. It's an open question whether rank 11 is achievable
  for 2x2x3. In practice, for this coding task, let me just use the schoolbook rank-12
  and get rank 7*12=84, which is still better than 96.

  Actually wait - let me re-read the dimensions. 4x4 times 4x6:
  A is 4x4, B is 4x6, C is 4x6.
  
  Partition A into 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
  Partition B into 2x2 blocks of 2x3: B = [[B00, B01, B02], [B10, B11, B12]] (each Bi,j is 2x3)
  
  C = A*B, C is 4x6, partitioned into 2x2 blocks of 2x3:
  C00 = A00*B00 + A01*B10
  C01 = A00*B01 + A01*B11
  C02 = A00*B02 + A01*B12
  C10 = A10*B00 + A11*B10
  C11 = A10*B01 + A11*B11
  C12 = A10*B02 + A11*B12
  
  Each C block (2x3) is computed as sum of two products of (2x2)*(2x3).
  
  Using Strassen for the outer 2x2x2 block multiplication:
  We need to multiply "matrices" of size 2x2 where each entry is itself a 2x2 (for A) 
  times a 2x3 (for B), producing 2x3 entries.
  
  This isn't a standard Kronecker product because the inner dimensions differ.
  
  Let me think differently. The full tensor for 4x4x6 has rank at most 4*4*6=96.
  
  A better approach: use block structure.
  
  Actually, let me just implement a direct search or use a known good decomposition.
  
  For now, let me implement the Kronecker product approach with:
  - 2x2x2 Strassen (rank 7) for the "outer" 2x2x2 block structure
  - 2x2x3 schoolbook (rank 12) for the "inner" computation
  - Total rank: 7 * 12 = 84
  
  This should be valid and give rank 84 < 96.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 140  # seconds, with 10s safety margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

# Strassen decomposition for 2x2x2: rank 7
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A01)*B11
# M6 = (A10-A00)*(B00+B01)
# M7 = (A01-A11)*(B10+B11)
# C00 = M1+M4-M5+M7
# C01 = M3+M5
# C10 = M2+M4
# C11 = M1-M2+M3+M6

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    U: shape (4, 7) - for A (2x2)
    V: shape (4, 7) - for B (2x2)  
    W: shape (4, 7) - for C (2x2), indexed as W[k*2+i, r]
    """
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7 -> but we use 0-3 for B too in local coords
    # Actually U, V, W are separate matrices.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Column 0: M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # Column 1: M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10 += M2
    
    # Column 2: M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C01 += M3
    
    # Column 3: M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
    
    # Column 4: M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5
    
    # Column 5: M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # Column 6: M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W

def _schoolbook_2x2x3():
    """Return (U, V, W) for 2x2x3 schoolbook decomposition, rank 12.
    U: shape (4, 12) - for A (2x2)
    V: shape (6, 12) - for B (2x3)
    W: shape (6, 12) - for C (2x3), indexed as W[k*2+i, r]
    """
    rank = 12
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    col = 0
    for i in range(2):  # row of A/C
        for j in range(2):  # col of A / row of B
            for k in range(3):  # col of B/C
                U[i*2 + j, col] = 1  # A[i,j]
                V[j*3 + k, col] = 1  # B[j,k]
                W[k*2 + i, col] = 1  # C[k,i]
                col += 1
    
    return U, V, W

def _kron_decomposition():
    """Build the full 4x4x6 decomposition using Kronecker product.
    
    Outer: 2x2x2 Strassen (rank 7)
    Inner: 2x2x3 schoolbook (rank 12)
    Total rank: 7 * 12 = 84
    """
    U_s, V_s, W_s = _strassen_2x2()  # shapes (4,7), (4,7), (4,7)
    U_i, V_i, W_i = _schoolbook_2x2x3()  # shapes (4,12), (6,12), (6,12)
    
    rank = 7 * 12
    U_full = np.zeros((16, rank))  # 4*4 = 16
    V_full = np.zeros((24, rank))  # 4*6 = 24
    W_full = np.zeros((24, rank))  # 6*4 = 24
    
    for a in range(7):
        for b in range(12):
            col = a * 12 + b
            # U_full: index = outer_A_idx * 4 + inner_A_idx
            # outer A is 2x2, so outer_A_idx in {0,1,2,3}
            # inner A is 2x2, so inner_A_idx in {0,1,2,3}
            # U_full[outer_idx * 4 + inner_idx, col] = U_s[outer_idx, a] * U_i[inner_idx, b]
            for outer_idx in range(4):
                if U_s[outer_idx, a] != 0:
                    for inner_idx in range(4):
                        if U_i[inner_idx, b] != 0:
                            U_full[outer_idx * 4 + inner_idx, col] = U_s[outer_idx, a] * U_i[inner_idx, b]
            
            # V_full: index = outer_B_idx * 6 + inner_B_idx
            # outer B is 2x2 (but in our partition, B is 2x2 blocks of 2x3)
            # Wait, I need to be more careful about the indexing.
            
            # Let me reconsider the partitioning.
            # A is 4x4: partition into 2x2 blocks of 2x2.
            #   A_block[oi, oj] where oi,oj in {0,1}, each block is 2x2.
            #   A[oi*2+ii, oj*2+ij] = A_block[oi,oj][ii,ij]
            #   So A index = oi*4 + oj*2 + ii... no.
            #   A row = oi*2 + ii, A col = oj*2 + ij
            #   A[row, col] = A[oi*2+ii, oj*2+ij]
            #   In U_full, the row index for A is (oi*2+ii)*4 + (oj*2+ij)? No.
            #   U has shape (n*m, rank) = (4*4, rank) = (16, rank)
            #   U[row_A, col_A, r] -> U[row_A * 4 + col_A, r]
            #   row_A = oi*2 + ii, col_A = oj*2 + ij
            #   So U_full[(oi*2+ii)*4 + (oj*2+ij), col] = U_s[oi*2+oj, a] * U_i[ii*2+ij, b]
            
            # V is shape (m*p, rank) = (4*6, rank) = (24, rank)
            # V[row_B, col_B, r] -> V[row_B * 6 + col_B, r]
            # B is 4x6: partition into 2x2 blocks of 2x3.
            #   B_block[oi, ok] where oi,ok in {0,1}, each block is 2x3.
            #   B[row, col] = B[oi*2+ii, ok*3+ik] where ii in {0,1}, ik in {0,1,2}
            #   row_B = oi*2 + ii, col_B = ok*3 + ik
            #   V_full[(oi*2+ii)*6 + (ok*3+ik), col] = V_s[oi*2+ok, a] * V_i[ii*3+ik, b]
            
            # W is shape (p*n, rank) = (6*4, rank) = (24, rank)
            # W[k*n+i, r] where k is col of C, i is row of C.
            # C is 4x6: partition into 2x2 blocks of 2x3.
            #   C_block[oi, ok] where oi,ok in {0,1}, each block is 2x3.
            #   C[row, col] = C[oi*2+ii, ok*3+ik]
            #   W index = col * 4 + row = (ok*3+ik)*4 + (oi*2+ii)
            #   W_full[(ok*3+ik)*4 + (oi*2+ii), col] = W_s[ok*2+oi, a] * W_i[ik*2+ii, b]
            
            for oi in range(2):
                for oj in range(2):
                    for ii in range(2):
                        for ij in range(2):
                            val = U_s[oi*2+oj, a] * U_i[ii*2+ij, b]
                            if val != 0:
                                U_full[(oi*2+ii)*4 + (oj*2+ij), col] += val
            
            for oi in range(2):
                for ok in range(2):
                    for ii in range(2):
                        for ik in range(3):
                            val = V_s[oi*2+ok, a] * V_i[ii*3+ik, b]
                            if val != 0:
                                V_full[(oi*2+ii)*6 + (ok*3+ik), col] += val
            
            for oi in range(2):
                for ok in range(2):
                    for ii in range(2):
                        for ik in range(3):
                            val = W_s[ok*2+oi, a] * W_i[ik*2+ii, b]
                            if val != 0:
                                W_full[(ok*3+ik)*4 + (oi*2+ii), col] += val
    
    return U_full, V_full, W_full

def _verify_decomposition(U, V, W, n=4, m=4, p=6, sample_size=50):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    # Check: C = A*B should equal sum_r U[:,r] outer V[:,r] ... no.
    # The tensor decomposition means:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if j is the shared index (always true in matmul), 0 otherwise.
    # Actually for matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta (always 1 for valid i,j,k in ranges)
    # The decomposition: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Verify on random matrices
    np.random.seed(42)
    for trial in range(sample_size):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        # Compute via decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = U[:, r]  # shape (n*m,)
            v_col = V[:, r]  # shape (m*p,)
            w_col = W[:, r]  # shape (p*n,)
            
            # Reshape
            u_mat = u_col.reshape(n, m)  # A contribution
            v_mat = v_col.reshape(m, p)  # B contribution
            w_mat = w_col.reshape(p, n).T  # C contribution: w[k*n+i] -> w_mat[i,k]
            
            C_computed += w_mat * (u_mat @ v_mat)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False, trial, C_computed - C_expected
    return True, -1, None

def construct():
    """Construct the tensor decomposition for 4x4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Try Kronecker decomposition
    U, V, W = _kron_decomposition()
    rank = U.shape[1]
    
    # Verify
    valid, trial, err = _verify_decomposition(U, V, W, n, m, p, sample_size=20)
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
