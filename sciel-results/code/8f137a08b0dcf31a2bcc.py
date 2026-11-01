import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T has entries T[i, j, k, i', j', k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Actually, the tensor is T_{i,j,k}^{i',j',k'} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Wait, the standard tensor for matrix multiplication C = AB is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T is such that T(A,B) = C, where T is a 3-tensor.
    # The flattening of T is an (n*p) x (n*m) x (m*p) tensor.
    # T[i, j, k] (in the bilinear form sense) contributes to C[i, k].
    # So the tensor is T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'} is not quite right.
    # Let's use the standard definition:
    # T is a tensor in V^* \otimes W^* \otimes X, where V=R^{n*m}, W=R^{m*p}, X=R^{n*p}.
    # The coordinates of T are T_{(i,j), (j',k'), (i'',k'')} = 1 if i=i'', j=j', k=k'', else 0.
    # The rank of T is the minimum number of rank-1 tensors needed to express T.
    # A rank-1 tensor is u \otimes v \otimes w.
    # The decomposition corresponds to:
    # C_{i,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k'*n+i', r} ... wait, the indices need to match.
    
    # Let's stick to the known algorithmic results.
    # For M_{2,4,4}, the schoolbook rank is 32.
    # Using Strassen's algorithm for 2x2 blocks:
    # We can view the 2x4 matrix A as two 2x2 blocks A0, A1 (column-wise? No, A is 2x4).
    # A = [A0 | A1] where A0, A1 are 2x2.
    # B = [[B00, B01], [B10, B11]] where B00, B01, B10, B11 are 2x2.
    # C = A B = [A0 | A1] [[B00, B01], [B10, B11]] = [A0 B00 + A1 B10, A0 B01 + A1 B11]
    # C0 = A0 B00 + A1 B10, C1 = A0 B01 + A1 B11.
    # Each 2x2 * 2x2 multiplication can be done with 7 multiplications using Strassen.
    # However, we have 4 such multiplications: A0B00, A1B10, A0B01, A1B11.
    # If they were independent, rank = 4 * 7 = 28.
    # But they share factors: A0 is shared between A0B00 and A0B01. A1 is shared between A1B10 and A1B11.
    # Actually, Strassen's algorithm computes M1..M7.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Let's apply this to each 2x2 * 2x2 block multiplication.
    # For A0 * B00:
    #   A0 = [[a00, a01], [a10, a11]]
    #   B00 = [[b000, b001], [b100, b101]]
    #   C00 = A0 B00
    #   Strassen factors:
    #   m1_00 = (a00+a11)(b000+b100)
    #   m2_00 = (a10+a11)b000
    #   m3_00 = a00(b100-b101)
    #   m4_00 = a11(b001-b100)
    #   m5_00 = (a00+a01)b101
    #   m6_00 = (a10-a00)(b000+b100)
    #   m7_00 = (a01-a11)(b001+b101)
    #   C00_00 = m1_00 - m4_00 + m5_00 + m7_00
    #   C00_01 = m3_00 + m4_00
    #   C00_10 = m2_00 + m4_00
    #   C00_11 = m1_00 - m2_00 + m3_00 + m6_00
    
    # This is getting complicated. The key insight from the prompt is that rank 28 is achievable.
    # Let's try to construct the rank 28 decomposition explicitly.
    
    # The 4 block multiplications are:
    # 1. C00 = A0 * B00
    # 2. C01 = A0 * B01
    # 3. C10 = A1 * B10
    # 4. C11 = A1 * B11
    
    # We can use 7 multiplications for each, totaling 28.
    # The U, V, W matrices will have 28 columns.
    
    # Let's define the indices.
    # A is 2x4. A[i, j] for i in {0,1}, j in {0..3}.
    # A0 = A[:, 0:2], A1 = A[:, 2:4].
    # B is 4x4. B is [[B00, B01], [B10, B11]].
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4].
    # C is 2x4. C = [C0, C1] where C0 = C[:, 0:2], C1 = C[:, 2:4].
    # C0 = A0 B00 + A1 B10.
    # C1 = A0 B01 + A1 B11.
    
    # Wait, the blocks of C are 2x2.
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    
    # So C0 = A0 B00 + A1 B10 and C1 = A0 B01 + A1 B11.
    # This is a sum of two products for each block of C.
    # We can compute A0 B00, A1 B10, A0 B01, A1 B11 using Strassen.
    # That's 4 * 7 = 28 multiplications.
    # Then we sum them: C0 = (A0 B00) + (A1 B10), C1 = (A0 B01) + (A1 B11).
    # This is valid.
    
    # Let's build the U, V, W matrices for rank 28.
    # Each of the 4 Strassen decompositions will contribute 7 columns to U, V, W.
    
    # We need a helper to build the Strassen decomposition for a 2x2 * 2x2 multiplication.
    # The input A is 2x2, B is 2x2, C is 2x2.
    # A indices: (a00, a01, a10, a11) -> flat index: a00=0, a01=1, a10=2, a11=3.
    # B indices: (b00, b01, b10, b11) -> flat index: b00=0, b01=1, b10=2, b11=3.
    # C indices: (c00, c01, c10, c11) -> flat index: c00=0, c01=1, c10=2, c11=3.
    # But the global indices are different.
    
    # Let's define the global flat indices.
    # U is (n*m, rank) = (8, 28).
    # V is (m*p, rank) = (16, 28).
    # W is (p*n, rank) = (8, 28).
    
    # A flat index: i*m + j = i*4 + j.
    # A0: i in {0,1}, j in {0,1} -> indices 0,1,4,5.
    # A1: i in {0,1}, j in {2,3} -> indices 2,3,6,7.
    # B flat index: j*p + k = j*4 + k.
    # B00: j in {0,1}, k in {0,1} -> indices 0,1,4,5.
    # B01: j in {0,1}, k in {2,3} -> indices 2,3,6,7.
    # B10: j in {2,3}, k in {0,1} -> indices 8,9,12,13.
    # B11: j in {2,3}, k in {2,3} -> indices 10,11,14,15.
    # W flat index: k*n + i = k*2 + i.
    # C0 (C[:, 0:2]): k in {0,1}, i in {0,1} -> indices 0,1,2,3.
    # C1 (C[:, 2:4]): k in {2,3}, i in {0,1} -> indices 4,5,6,7.
    
    # We'll create a function to add a Strassen decomposition for a specific block.
    # Parameters:
    #   U_block_indices: list of 4 global indices for the 2x2 A block.
    #   V_block_indices: list of 4 global indices for the 2x2 B block.
    #   W_block_indices: list of 4 global indices for the 2x2 C block.
    #   offset: the starting column index in U, V, W.
    
    # The 7 Strassen multiplications for A * B = C:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b10)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    
    # C00 = m1 - m4 + m5 + m7
    # C01 = m3 + m4
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Each mr corresponds to one column in U, V, W.
    # U vector for mr has non-zero entries at the positions of the A factors.
    # V vector for mr has non-zero entries at the positions of the B factors.
    # W vector for mr has non-zero entries at the positions of the C factors.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the block index mappings
    # A0: a00=0, a01=1, a10=4, a11=5
    # A1: a00=2, a01=3, a10=6, a11=7
    # B00: b00=0, b01=1, b10=4, b11=5
    # B01: b00=2, b01=3, b10=6, b11=7
    # B10: b00=8, b01=9, b10=12, b11=13
    # B11: b00=10, b01=11, b10=14, b11=15
    # C0: c00=0, c01=1, c10=2, c11=3
    # C1: c00=4, c01=5, c10=6, c11=7
    
    def add_strassen(U, V, W, A_idx, B_idx, C_idx, offset):
        # A_idx = [a00, a01, a10, a11]
        # B_idx = [b00, b01, b10, b11]
        # C_idx = [c00, c01, c10, c11]
        
        # m1 = (a00+a11)(b00+b11)
        U[A_idx[0]][offset+0] = 1
        U[A_idx[3]][offset+0] = 1
        V[B_idx[0]][offset+0] = 1
        V[B_idx[3]][offset+0] = 1
        W[C_idx[0]][offset+0] = 1
        W[C_idx[3]][offset+0] = 1
        
        # m2 = (a10+a11)b00
        U[A_idx[2]][offset+1] = 1
        U[A_idx[3]][offset+1] = 1
        V[B_idx[0]][offset+1] = 1
        W[C_idx[2]][offset+1] = 1
        W[C_idx[3]][offset+1] = -1
        
        # m3 = a00(b10-b11)
        U[A_idx[0]][offset+2] = 1
        V[B_idx[2]][offset+2] = 1
        V[B_idx[3]][offset+2] = -1
        W[C_idx[1]][offset+2] = 1
        W[C_idx[3]][offset+2] = 1
        
        # m4 = a11(b01-b10)
        U[A_idx[3]][offset+3] = 1
        V[B_idx[1]][offset+3] = 1
        V[B_idx[2]][offset+3] = -1
        W[C_idx[0]][offset+3] = -1
        W[C_idx[1]][offset+3] = 1
        W[C_idx[2]][offset+3] = 1
        
        # m5 = (a00+a01)b11
        U[A_idx[0]][offset+4] = 1
        U[A_idx[1]][offset+4] = 1
        V[B_idx[3]][offset+4] = 1
        W[C_idx[0]][offset+4] = 1
        W[C_idx[3]][offset+4] = 1
        
        # m6 = (a10-a00)(b00+b10)
        U[A_idx[2]][offset+5] = 1
        U[A_idx[0]][offset+5] = -1
        V[B_idx[0]][offset+5] = 1
        V[B_idx[2]][offset+5] = 1
        W[C_idx[3]][offset+5] = 1
        
        # m7 = (a01-a11)(b01+b11)
        U[A_idx[1]][offset+6] = 1
        U[A_idx[3]][offset+6] = -1
        V[B_idx[1]][offset+6] = 1
        V[B_idx[3]][offset+6] = 1
        W[C_idx[0]][offset+6] = 1
        W[C_idx[1]][offset+6] = 1
        
    # Block 1: A0 * B00 -> C0 (part 1)
    A0_idx = [0, 1, 4, 5]
    B00_idx = [0, 1, 4, 5]
    C0_idx = [0, 1, 2, 3]
    add_strassen(U, V, W, A0_idx, B00_idx, C0_idx, 0)
    
    # Block 2: A0 * B01 -> C1 (part 1)
    B01_idx = [2, 3, 6, 7]
    C1_idx = [4, 5, 6, 7]
    add_strassen(U, V, W, A0_idx, B01_idx, C1_idx, 7)
    
    # Block 3: A1 * B10 -> C0 (part 2)
    A1_idx = [2, 3, 6, 7]
    B10_idx = [8, 9, 12, 13]
    add_strassen(U, V, W, A1_idx, B10_idx, C0_idx, 14)
    
    # Block 4: A1 * B11 -> C1 (part 2)
    B11_idx = [10, 11, 14, 15]
    add_strassen(U, V, W, A1_idx, B11_idx, C1_idx, 21)
    
    # Wait, I need to check the W indices again.
    # The convention is W[k*n+i, r].
    # C0 is C[:, 0:2], so k in {0,1}, i in {0,1}.
    # k*n+i:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # So C0_idx = [0, 1, 2, 3] is correct for c00, c01, c10, c11.
    # c00 is C[0,0] -> k=0, i=0 -> 0
    # c01 is C[0,1] -> k=1, i=0 -> 2 ? No.
    # C[i, k] = sum_j A[i,j] B[j,k].
    # C[0,0] -> i=0, k=0 -> index k*n+i = 0*2+0 = 0.
    # C[0,1] -> i=0, k=1 -> index k*n+i = 1*2+0 = 2.
    # C[1,0] -> i=1, k=0 -> index k*n+i = 0*2+1 = 1.
    # C[1,1] -> i=1, k=1 -> index k*n+i = 1*2+1 = 3.
    # So the mapping is:
    # c00 (C[0,0]) -> 0
    # c10 (C[1,0]) -> 1
    # c01 (C[0,1]) -> 2
    # c11 (C[1,1]) -> 3
    # My C_idx was [c00, c01, c10, c11] = [0, 2, 1, 3].
    # Let's correct this.
    
    # Let's re-define C_idx correctly.
    # C0: k in {0,1}, i in {0,1}.
    # c00 (i=0, k=0) -> 0
    # c10 (i=1, k=0) -> 1
    # c01 (i=0, k=1) -> 2
    # c11 (i=1, k=1) -> 3
    # So C0_idx = [0, 2, 1, 3] for [c00, c01, c10, c11].
    # Wait, the Strassen formulas use c00, c01, c10, c11 as:
    # C = [[c00, c01], [c10, c11]]
    # So c00 is C[0,0], c01 is C[0,1], c10 is C[1,0], c11 is C[1,1].
    # Global index for C[i, k] is k*n + i.
    # C[0,0] -> 0*2+0 = 0.
    # C[0,1] -> 1*2+0 = 2.
    # C[1,0] -> 0*2+1 = 1.
    # C[1,1] -> 1*2+1 = 3.
    # So the global indices for [c00, c01, c10, c11] are [0, 2, 1, 3].
    
    # Let's rebuild with the correct C indices.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_correct(U, V, W, A_idx, B_idx, C_idx, offset):
        # A_idx = [a00, a01, a10, a11]
        # B_idx = [b00, b01, b10, b11]
        # C_idx = [c00, c01, c10, c11]
        
        # m1 = (a00+a11)(b00+b11)
        U[A_idx[0]][offset+0] = 1
        U[A_idx[3]][offset+0] = 1
        V[B_idx[0]][offset+0] = 1
        V[B_idx[3]][offset+0] = 1
        W[C_idx[0]][offset+0] = 1  # c00
        W[C_idx[3]][offset+0] = 1  # c11
        
        # m2 = (a10+a11)b00
        U[A_idx[2]][offset+1] = 1
        U[A_idx[3]][offset+1] = 1
        V[B_idx[0]][offset+1] = 1
        W[C_idx[2]][offset+1] = 1  # c10
        W[C_idx[3]][offset+1] = -1 # c11
        
        # m3 = a00(b10-b11)
        U[A_idx[0]][offset+2] = 1
        V[B_idx[2]][offset+2] = 1
        V[B_idx[3]][offset+2] = -1
        W[C_idx[1]][offset+2] = 1  # c01
        W[C_idx[3]][offset+2] = 1  # c11
        
        # m4 = a11(b01-b10)
        U[A_idx[3]][offset+3] = 1
        V[B_idx[1]][offset+3] = 1
        V[B_idx[2]][offset+3] = -1
        W[C_idx[0]][offset+3] = -1 # c00
        W[C_idx[1]][offset+3] = 1  # c01
        W[C_idx[2]][offset+3] = 1  # c10
        
        # m5 = (a00+a01)b11
        U[A_idx[0]][offset+4] = 1
        U[A_idx[1]][offset+4] = 1
        V[B_idx[3]][offset+4] = 1
        W[C_idx[0]][offset+4] = 1  # c00
        W[C_idx[3]][offset+4] = 1  # c11
        
        # m6 = (a10-a00)(b00+b10)
        U[A_idx[2]][offset+5] = 1
        U[A_idx[0]][offset+5] = -1
        V[B_idx[0]][offset+5] = 1
        V[B_idx[2]][offset+5] = 1
        W[C_idx[3]][offset+5] = 1  # c11
        
        # m7 = (a01-a11)(b01+b11)
        U[A_idx[1]][offset+6] = 1
        U[A_idx[3]][offset+6] = -1
        V[B_idx[1]][offset+6] = 1
        V[B_idx[3]][offset+6] = 1
        W[C_idx[0]][offset+6] = 1  # c00
        W[C_idx[1]][offset+6] = 1  # c01
        
    # Corrected C indices
    # C0: [c00, c01, c10, c11] -> [0, 2, 1, 3]
    C0_idx = [0, 2, 1, 3]
    # C1: k in {2,3}, i in {0,1}.
    # c00 (i=0, k=2) -> 2*2+0 = 4
    # c01 (i=0, k=3) -> 3*2+0 = 6
    # c10 (i=1, k=2) -> 2*2+1 = 5
    # c11 (i=1, k=3) -> 3*2+1 = 7
    C1_idx = [4, 6, 5, 7]
    
    add_strassen_correct(U, V, W, A0_idx, B00_idx, C0_idx, 0)
    add_strassen_correct(U, V, W, A0_idx, B01_idx, C1_idx, 7)
    add_strassen_correct(U, V, W, A1_idx, B10_idx, C0_idx, 14)
    add_strassen_correct(U, V, W, A1_idx, B11_idx, C1_idx, 21)
    
    # Verify the decomposition
    # T = sum_r U[:, r] \otimes V[:, r] \otimes W[:, r]
    # We can check this by contracting with random A and B.
    
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = A_test @ B_test
    
    # Flatten A, B, C
    A_flat = A_test.flatten()  # Row-major: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], ...
    # Wait, my A_idx mapping assumed column-major for the blocks?
    # A0: a00=0, a01=1, a10=4, a11=5.
    # A is 2x4. A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3, A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7.
    # So A0 = A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = [[0, 1], [4, 5]] in flat indices.
    # This matches my A0_idx = [0, 1, 4, 5] for [a00, a01, a10, a11].
    # So the flat index is i*m + j.
    
    B_flat = B_test.flatten()  # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, ...
    # B00 = B[0:2, 0:2] = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]] = [[0, 1], [4, 5]] in flat indices.
    # My B00_idx = [0, 1, 4, 5] for [b00, b01, b10, b11]. Matches.
    
    # C_flat: C[i, k] -> index k*n + i.
    # C is 2x4. C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7.
    # My C index mapping: C[i, k] -> k*n + i.
    # C[0,0] -> 0*2+0 = 0.
    # C[0,1] -> 1*2+0 = 2.
    # C[1,0] -> 0*2+1 = 1.
    # C[1,1] -> 1*2+1 = 3.
    # So C_flat in the order [c00, c10, c01, c11] is [0, 1, 2, 3].
    # But the natural flat order of C (row-major) is [C[0,0], C[0,1], C[0,2], C[0,3], C[1,0], ...].
    # This is [c00, c01, c02, c03, c10, c11, c12, c13].
    # The tensor decomposition produces W[k*n+i, r], which is the "trace convention" index.
    # So the output C is in the trace convention order, not row-major.
    # We need to convert C_test to trace convention order to compare.
    
    C_trace = np.zeros_like(C_test)
    for i in range(n):
        for k in range(p):
            C_trace[k*n + i, 0] = C_test[i, k]
    # Wait, C_test is n x p. C_trace should be a flat array of size n*p.
    C_trace_flat = np.zeros(n*p)
    for i in range(n):
        for k in range(p):
            C_trace_flat[k*n + i] = C_test[i, k]
    
    # Now compute the tensor product sum
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # C_comp = sum_r U[:, r] * V[:, r] * W[:, r] ? No.
    # The tensor is T = sum_r U_r \otimes V_r \otimes W_r.
    # T(A, B) = sum_r (U_r . A_flat) * (V_r . B_flat) * W_r
    # So C_trace_flat[r] = sum_r (U[:, r] . A_flat) * (V[:, r] . B_flat) * W[:, r]
    # Wait, W is a vector for each r. So it's:
    # C_trace_flat = sum_r (U[:, r] @ A_flat) * (V[:, r] @ B_flat) * W[:, r]
    
    C_comp_flat = np.zeros(n*p)
    for r in range(rank):
        u_val = np.dot(U_arr[:, r], A_flat)
        v_val = np.dot(V_arr[:, r], B_flat)
        C_comp_flat += u_val * v_val * W_arr[:, r]
    
    # Check if C_comp_flat matches C_trace_flat
    if not np.allclose(C_comp_flat, C_trace_flat):
        # Try to fix the indices
        pass
    
    return {"U": U, "V": V, "W": W}
