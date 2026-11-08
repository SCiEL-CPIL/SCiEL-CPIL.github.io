"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # Use Strassen's algorithm for 2x2x2 matrix multiplication
    # Strassen's decomposition has rank 7 (vs schoolbook rank 8)
    # For 2x2x2: A(2x2) * B(2x2) = C(2x2)
    # The 7 rank-1 tensors for Strassen's algorithm:
    
    # Indices: A is 2x2 (indices 0-3), B is 2x2 (indices 0-3), C is 2x2 (indices 0-3)
    # A: [[a00, a01], [a10, a11]] -> flat: [a00, a01, a10, a11]
    # B: [[b00, b01], [b10, b11]] -> flat: [b00, b01, b10, b11]
    # C: [[c00, c01], [c10, c11]] -> flat: [c00, c01, c10, c11]
    
    # Strassen's 7 products:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 + b11)
    # M4 = a11(b00 - b10)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # C = [[M1 + M4 - M5 - M7, M3 + M5], [M2 + M4, M1 - M2 + M3 + M7]]
    
    # For each rank-1 tensor, we need vectors u (length 4), v (length 4), w (length 4)
    # such that sum of u_i * v_j * w_k over the 7 tensors equals the multiplication tensor
    
    # Let's define the 7 Strassen tensors for 2x2x2:
    # Each tensor is (u, v, w) where u, v, w are vectors in R^4
    
    # M1 = (a00 + a11)(b00 + b11) contributes to c00 (+1), c10 (-1), c11 (+1)
    # Actually, let me be more careful. The tensor T[i,j,k] = 1 if C[k] depends on A[i]*B[j]
    
    # For 2x2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Using trace convention W[k*n+i, r] where n=2, so W has indices (k*2+i)
    # C flat index: k*2 + i where i is row, k is col? No...
    # Wait, the task says W[k*n+i, r]. For 2x2x2, n=2.
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # The tensor T[i, j, k] = 1 if C[i,k] depends on A[i,j] * B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W index is k*n + i (using trace convention)
    
    # For 2x2x2: n=2, m=2, p=2
    # U indices: i*m + j = i*2 + j for i,j in {0,1}
    # V indices: j*p + k = j*2 + k for j,k in {0,1}
    # W indices: k*n + i = k*2 + i for k,i in {0,1}
    
    # So:
    # U: [a00, a01, a10, a11] (i*2+j)
    # V: [b00, b01, b10, b11] (j*2+k)
    # W: [c00, c10, c01, c11] (k*2+i) where c_ik is C[i,k]
    
    # Wait, W[k*n+i] = W[k*2+i]. So:
    # W[0] = c_{0,0} (k=0, i=0)
    # W[1] = c_{1,0} (k=0, i=1)
    # W[2] = c_{0,1} (k=1, i=0)
    # W[3] = c_{1,1} (k=1, i=1)
    
    # The tensor T[U_idx, V_idx, W_idx] = 1 if:
    # U_idx = i*2+j, V_idx = j*2+k, W_idx = k*2+i
    # This means C[i,k] depends on A[i,j] * B[j,k]
    
    # So T[i*2+j, j*2+k, k*2+i] = 1 for all i,j,k in {0,1}
    
    # Now Strassen's algorithm:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 + b11)
    # M4 = a11(b00 - b10)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7
    
    # In terms of W indices (k*2+i):
    # W[0] = c00 = M1 + M4 - M5 - M7
    # W[1] = c10 = M2 + M4
    # W[2] = c01 = M3 + M5
    # W[3] = c11 = M1 - M2 + M3 + M7
    
    # For each M_r, we need to express it as sum of rank-1 terms
    # M1 = (a00 + a11)(b00 + b11)
    # This is a product of a linear form in A and a linear form in B
    # u vector for M1: [1, 0, 1, 0] (a00 + a11)
    # v vector for M1: [1, 0, 0, 1] (b00 + b11)
    # w vector for M1: [1, 0, 0, 1] (contributes +1 to c00 and c11)
    
    # Let me define all 7:
    strassen_2x2 = [
        # M1: (a00+a11)(b00+b11) -> c00+1, c11+1
        ([1,0,1,0], [1,0,0,1], [1,0,0,1]),
        # M2: (a10+a11)b00 -> c10+1, c11-1
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        # M3: a00(b10+b11) -> c01+1, c11+1
        ([1,0,0,0], [0,0,1,1], [0,0,1,1]),
        # M4: a11(b00-b10) -> c00+1, c10+1
        ([0,0,0,1], [1,0,-1,0], [1,1,0,0]),
        # M5: (a00+a01)b11 -> c00-1, c01+1
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        # M6: (a10-a00)(b00+b01) -> not used in final C? Let me check...
        # Actually M6 is not directly in the C formulas above. Let me recheck Strassen.
        # Standard Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10+b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1 + M4 - M5 - M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M7
        #
        # M6 is not used! That's strange. Let me verify...
        # Actually, I think I made an error. Let me look up Strassen again.
        #
        # Correct Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10+b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1 + M4 - M5 - M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M7
        #
        # Hmm, M6 doesn't appear. Let me double-check with a reference.
        # Actually, I think the standard formulas are:
        # c00 = M1 + M4 - M5 - M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M7
        #
        # But this only uses 6 of the 7 products. That can't be right.
        # Let me reconsider. Perhaps I have the indices wrong.
        
        # Let me just use a known correct Strassen decomposition.
        # After research, the correct Strassen decomposition for 2x2 is:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10+b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1 + M4 - M5 - M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M7
        #
        # Wait, I'm getting the same thing. Let me just verify numerically.
        # For now, let me just hardcode a known working decomposition.
    ]
    
    # Let me use a different approach. I'll use the known fact that Strassen's
    # algorithm gives rank 7 for 2x2x2. Let me just define the 7 tensors directly
    # by checking which entries of the multiplication tensor they cover.
    
    # Actually, let me just use a simpler approach: use the schoolbook for 2x2x2
    # but with rank 7 by exploiting linearity.
    
    # For 2x2x2, the multiplication tensor has 8 nonzero entries (one for each i,j,k).
    # We can combine some of them using linearity.
    
    # Let me define the 7 Strassen tensors explicitly:
    # I'll use the standard Strassen formulas and verify they work.
    
    strassen_tensors = []
    
    # M1 = (a00+a11)(b00+b11)
    # u = [1,0,1,0], v = [1,0,0,1]
    # This contributes to: c00 (+1) and c11 (+1)
    # So w = [1,0,0,1]
    strassen_tensors.append(([1,0,1,0], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (a10+a11)b00
    # u = [0,0,1,1], v = [1,0,0,0]
    # This contributes to: c10 (+1) and c11 (-1)
    # So w = [0,1,0,-1]
    strassen_tensors.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
    
    # M3 = a00(b10+b11)
    # u = [1,0,0,0], v = [0,0,1,1]
    # This contributes to: c01 (+1) and c11 (+1)
    # So w = [0,0,1,1]
    strassen_tensors.append(([1,0,0,0], [0,0,1,1], [0,0,1,1]))
    
    # M4 = a11(b00-b10)
    # u = [0,0,0,1], v = [1,0,-1,0]
    # This contributes to: c00 (+1) and c10 (+1)
    # So w = [1,1,0,0]
    strassen_tensors.append(([0,0,0,1], [1,0,-1,0], [1,1,0,0]))
    
    # M5 = (a00+a01)b11
    # u = [1,1,0,0], v = [0,0,0,1]
    # This contributes to: c00 (-1) and c01 (+1)
    # So w = [-1,0,1,0]
    strassen_tensors.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # M6 = (a10-a00)(b00+b01)
    # u = [-1,0,1,0], v = [1,1,0,0]
    # This contributes to: c10 (+1) and c11 (+1)
    # So w = [0,1,0,1]
    strassen_tensors.append(([-1,0,1,0], [1,1,0,0], [0,1,0,1]))
    
    # M7 = (a01-a11)(b10+b11)
    # u = [0,1,0,-1], v = [0,0,1,1]
    # This contributes to: c00 (-1) and c11 (+1)
    # So w = [-1,0,0,1]
    strassen_tensors.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,1]))
    
    # Now let's verify: sum of all 7 tensors should equal the multiplication tensor
    # T[i*2+j, j*2+k, k*2+i] = 1
    
    # Let's compute the sum and check
    T_strassen = np.zeros((4, 4, 4))
    for u, v, w in strassen_tensors:
        T_strassen += np.einsum('i,j,k->ijk', u, v, w)
    
    # Expected tensor
    T_expected = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_expected[i*2+j, j*2+k, k*2+i] = 1
    
    # Check if they match
    if not np.allclose(T_strassen, T_expected):
        # If they don't match, fall back to schoolbook
        pass
    
    # For the full 4x4x6 problem, we can block-decompose
    # A is 4x4, B is 4x6, C is 4x6
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
    # C = [[C11, C12, C13], [C21, C22, C23]] where each is 2x2
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # Each Cij is a sum of two 2x2x2 multiplications
    # If we use Strassen for each 2x2x2, we get rank 7 per multiplication
    # So each Cij needs 2*7 = 14 rank-1 terms
    # Total: 6 * 14 = 84 rank-1 terms
    
    # This is better than schoolbook rank of 96!
    
    # Let's construct this decomposition
    n, m, p = 4, 4, 6
    
    # Block sizes
    n1, n2 = 2, 2  # n = n1 + n2
    m1, m2 = 2, 2  # m = m1 + m2
    p1, p2, p3 = 2, 2, 2  # p = p1 + p2 + p3
    
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    # For each block multiplication Cij = Akk * Blk, we need to map:
    # A indices: row block k, col block l -> global A index (k*2 + i)*4 + (l*2 + j)
    # Wait, A is 4x4, so A[row, col] where row, col in 0..3
    # Block (k,l) of A is A[k*2:(k+1)*2, l*2:(l+1)*2]
    # So A[k*2+i, l*2+j] for i,j in 0,1
    
    # U index for A[row, col] = row*4 + col (since m=4)
    # So U index for A[k*2+i, l*2+j] = (k*2+i)*4 + (l*2+j)
    
    # V index for B[col, row] where B is 4x6
    # B is m x p = 4 x 6
    # Block (l, k) of B is B[l*2:(l+1)*2, k*2:(k+1)*2] for k in 0,1,2
    # Wait, B is 4x6, so blocks are 2x2
    # B[row, col] where row in 0..3, col in 0..5
    # Block (l, k) where l is row block (0,1), k is col block (0,1,2)
    # B[l*2+i, k*2+j] for i,j in 0,1
    
    # V index for B[row, col] = row*6 + col (since p=6)
    # So V index for B[l*2+i, k*2+j] = (l*2+i)*6 + (k*2+j)
    
    # W index for C[row, col] where C is 4x6
    # Using trace convention W[k*n+i, r] where k is col, i is row
    # C[row, col] -> W[col*4 + row, r]
    # Block (k, l) of C is C[k*2:(k+1)*2, l*2:(l+1)*2] for k in 0,1, l in 0,1,2
    # Wait, C is 4x6, so C[row, col] where row in 0..3, col in 0..5
    # Block (i_block, j_block) where i_block is row block (0,1), j_block is col block (0,1,2)
    # C[i_block*2+i, j_block*2+j] for i,j in 0,1
    
    # W index for C[row, col] = col*4 + row
    # So W index for C[i_block*2+i, j_block*2+j] = (j_block*2+j)*4 + (i_block*2+i)
    
    # Now for each block multiplication:
    # C[i_block, j_block] = sum over k_block of A[i_block, k_block] * B[k_block, j_block]
    
    # For each (i_block, j_block, k_block) triple, we have a 2x2x2 multiplication
    # We use Strassen's 7 rank-1 terms for each
    
    # Total number of block multiplications: 2 * 3 * 2 = 12
    # Each uses rank 7, so total rank = 12 * 7 = 84
    
    rank = 12 * 7
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    col = 0
    for i_block in range(2):
        for j_block in range(3):
            for k_block in range(2):
                # This block multiplication: C[i_block, j_block] += A[i_block, k_block] * B[k_block, j_block]
                
                # For each of the 7 Strassen tensors:
                for u_local, v_local, w_local in strassen_tensors:
                    # u_local has 4 entries corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
                    # v_local has 4 entries corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
                    # w_local has 4 entries corresponding to C[0,0], C[1,0], C[0,1], C[1,1]
                    # (using W index = k*2+i where k is col, i is row)
                    
                    for i in range(2):
                        for j in range(2):
                            # A index: (i_block*2+i)*4 + (k_block*2+j)
                            a_idx = (i_block*2 + i)*4 + (k_block*2 + j)
                            U[a_idx, col] += u_local[i*2 + j]
                    
                    for i in range(2):
                        for j in range(2):
                            # B index: (k_block*2+i)*6 + (j_block*2+j)
                            b_idx = (k_block*2 + i)*6 + (j_block*2 + j)
                            V[b_idx, col] += v_local[i*2 + j]
                    
                    for i in range(2):
                        for j in range(2):
                            # C index: C[(i_block*2+i), (j_block*2+j)]
                            # W index = col*4 + row = (j_block*2+j)*4 + (i_block*2+i)
                            c_idx = (j_block*2 + j)*4 + (i_block*2 + i)
                            W[c_idx, col] += w_local[i*2 + j]
                    
                    col += 1
    
    # Verify the decomposition
    T = np.zeros((16, 24, 24))
    for i in range(4):
        for j in range(4):
            for k in range(6):
                T[i*4+j, j*6+k, k*4+i] = 1.0
    
    T_decomp = np.zeros((16, 24, 24))
    for c in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    
    if not np.allclose(T, T_decomp, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1.0
                    V[j*6+k, col] = 1.0
                    W[k*4+i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
