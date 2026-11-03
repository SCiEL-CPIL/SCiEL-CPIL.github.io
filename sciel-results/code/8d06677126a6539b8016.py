"""Matrix multiplication 4,5,6: rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def _schoolbook_block(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m by m x p."""
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
    """Strassen decomposition for 2x2 by 2x2, rank 7.
    Inputs flattened in row-major: [a00,a01,a10,a11] for A, [b00,b01,b10,b11] for B.
    Output flattened: [c00,c01,c10,c11].
    """
    # 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _strassen_4x4():
    """Strassen decomposition for 4x4 by 4x4, rank 49.
    Uses 4 blocks of 2x2, each using 7 Strassen products.
    Input A: 4x4, flattened row-major (16 elements)
    Input B: 4x4, flattened row-major (16 elements)
    Output C: 4x4, flattened row-major (16 elements)
    """
    # Block structure: A = [[A00, A01], [A10, A11]], each 2x2
    # B = [[B00, B01], [B10, B11]], each 2x2
    # C = [[C00, C01], [C10, C11]]
    
    # Flatten indices:
    # A00: [0,1,4,5], A01: [2,3,6,7]
    # A10: [8,9,12,13], A11: [10,11,14,15]
    # B00: [0,1,4,5], B01: [2,3,6,7]
    # B10: [8,9,12,13], B11: [10,11,14,15]
    # C00: [0,1,4,5], C01: [2,3,6,7]
    # C10: [8,9,12,13], C11: [10,11,14,15]
    
    U00, V00, W00 = _strassen_2x2()
    
    # For each of the 7 Strassen products in the 4x4 decomposition,
    # we need to compute the 2x2 matrix products.
    # The 4x4 Strassen algorithm computes:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # Then C00 = P1+P4-P5+P7, etc.
    
    # But we need the rank-49 decomposition, which means we decompose each
    # 2x2 product into 7 scalar products, giving 7*7=49 total products.
    
    # Actually, the standard Strassen for 4x4 uses 7 recursive calls on 2x2,
    # and each 2x2 call has rank 7, so total rank is 7*7=49.
    
    # Let's build this carefully.
    # The 4x4 Strassen computes 7 intermediate 2x2 matrices, each via 7 scalar products.
    
    # P1 = (A00+A11)(B00+B11): 
    #   (A00+A11) is 2x2, (B00+B11) is 2x2
    #   U for A00+A11: rows 0,1,4,5 of A00 + rows 10,11,14,15 of A11
    #   V for B00+B11: cols 0,1,4,5 of B00 + cols 10,11,14,15 of B11
    
    # This is getting complex. Let me use a different approach.
    # I'll build the 4x4 Strassen by composing the 2x2 Strassen.
    
    # The 4x4 Strassen has 7 block products, each a 2x2*2x2 product.
    # Each 2x2*2x2 product has rank 7 (Strassen).
    # So total rank = 7 * 7 = 49.
    
    # For each of the 7 block products, we need to:
    # 1. Extract the appropriate linear combinations of A blocks
    # 2. Extract the appropriate linear combinations of B blocks
    # 3. Apply the 2x2 Strassen decomposition
    # 4. Map the output to the correct position in C
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Define block indices
    # A blocks: A00=[0,1,4,5], A01=[2,3,6,7], A10=[8,9,12,13], A11=[10,11,14,15]
    # B blocks: B00=[0,1,4,5], B01=[2,3,6,7], B10=[8,9,12,13], B11=[10,11,14,15]
    # C blocks: C00=[0,1,4,5], C01=[2,3,6,7], C10=[8,9,12,13], C11=[10,11,14,15]
    
    A00_idx = [0, 1, 4, 5]
    A01_idx = [2, 3, 6, 7]
    A10_idx = [8, 9, 12, 13]
    A11_idx = [10, 11, 14, 15]
    
    B00_idx = [0, 1, 4, 5]
    B01_idx = [2, 3, 6, 7]
    B10_idx = [8, 9, 12, 13]
    B11_idx = [10, 11, 14, 15]
    
    C00_idx = [0, 1, 4, 5]
    C01_idx = [2, 3, 6, 7]
    C10_idx = [8, 9, 12, 13]
    C11_idx = [10, 11, 14, 15]
    
    # The 7 Strassen products for 2x2:
    # For a 2x2 matrix X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # For each of the 7 block products in 4x4 Strassen, we apply the 2x2 Strassen.
    # Block product 1: (A00+A11)(B00+B11)
    # Block product 2: (A10+A11)B00
    # Block product 3: A00(B01-B11)
    # Block product 4: A11(B10-B00)
    # Block product 5: (A00+A01)B11
    # Block product 6: (A10-A00)(B00+B01)
    # Block product 7: (A01-A11)(B10+B11)
    
    # For each block product, we have a linear combination of A blocks and B blocks.
    # Then we apply the 2x2 Strassen to get 7 scalar products.
    
    # Let's define a helper to build the U, V, W for a block product.
    
    def build_block_product(a_combos, b_combos, c_target, col_offset):
        """
        a_combos: list of (block_indices, coefficient) for the 2x2 matrix X
        b_combos: list of (block_indices, coefficient) for the 2x2 matrix Y
        c_target: list of 4 indices for the output 2x2 block
        col_offset: starting column index
        """
        # X = sum of coeff * A_block for (block_indices, coeff) in a_combos
        # Y = sum of coeff * B_block for (block_indices, coeff) in b_combos
        
        # For each of the 7 Strassen products in the 2x2 decomposition:
        for s in range(7):
            col = col_offset + s
            # The 2x2 Strassen U, V, W for this product
            u2 = U00[:, s]
            v2 = V00[:, s]
            w2 = W00[:, s]
            
            # For each row of u2 (corresponding to x00, x01, x10, x11):
            for ri in range(4):
                if u2[ri] != 0:
                    for (block_idx, coeff) in a_combos:
                        for local_idx in range(4):
                            # local_idx maps to x00=0, x01=1, x10=2, x11=3
                            # block_idx is a list of 4 global indices
                            # The local position within the block:
                            # x00 -> block_idx[0], x01 -> block_idx[1], x10 -> block_idx[2], x11 -> block_idx[3]
                            global_idx = block_idx[ri // 2 + (ri % 2) * 4]
                            U[global_idx, col] += u2[ri] * coeff
            
            # For each row of v2 (corresponding to y00, y01, y10, y11):
            for ri in range(4):
                if v2[ri] != 0:
                    for (block_idx, coeff) in b_combos:
                        global_idx = block_idx[ri // 2 + (ri % 2) * 4]
                        V[global_idx, col] += v2[ri] * coeff
            
            # For each row of w2 (corresponding to c00, c01, c10, c11):
            for ri in range(4):
                if w2[ri] != 0:
                    W[c_target[ri], col] += w2[ri]
    
    # Block product 1: (A00+A11)(B00+B11)
    # X = A00 + A11, Y = B00 + B11
    # Output goes to C00 (with +1), C01 (with 0), C10 (with 0), C11 (with -1)
    # Wait, C00 = P1+P4-P5+P7, C01 = P3+P5, C10 = P2+P4, C11 = P1-P2+P3+P6
    # So P1 contributes to C00 with +1 and C11 with +1.
    
    # Actually, let me reconsider. The 4x4 Strassen computes:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    
    # Each Pi is a 2x2 matrix. So we need to add the appropriate Pi to each C block.
    
    # Let's build each Pi separately and then add them to the correct C blocks.
    
    # I'll rebuild this more carefully.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    col = 0
    
    # P1 = (A00+A11)(B00+B11)
    # X = A00 + A11, Y = B00 + B11
    for s in range(7):
        for ri in range(4):
            if U00[ri, s] != 0:
                for idx in A00_idx:
                    U[idx, col] += U00[ri, s]
                for idx in A11_idx:
                    U[idx, col] += U00[ri, s]
            if V00[ri, s] != 0:
                for idx in B00_idx:
                    V[idx, col] += V00[ri, s]
                for idx in B11_idx:
                    V[idx, col] += V00[ri, s]
        for ri in range(4):
            if W00[ri, s] != 0:
                W[C00_idx[ri], col] += W00[ri, s]
                W[C11_idx[ri], col] += W00[ri, s]
        col += 1
    
    # P2 = (A10+A11)B00
    for s in range(7):
        for ri in range(4):
            if U00[ri, s] != 0:
                for idx in A10_idx:
                    U[idx, col] += U00[ri, s]
                for idx in A11_idx:
                    U[idx, col] += U00[ri, s]
            if V00[ri, s] != 0:
                for idx in B00_idx:
                    V[idx, col] += V00[ri, s]
        for ri in range(4):
            if W00[ri, s] != 0:
                W[C10_idx[ri], col] += W00[ri, s]
                W[C11_idx[ri], col] -= W00[ri, s]
        col += 1
    
    # P3 = A00(B01-B11)
    for s in range(7):
        for ri in range(4):
            if U00[ri, s] != 0:
                for idx in A00_idx:
                    U[idx, col] += U00[ri, s]
            if V00[ri, s] != 0:
                for idx in B01_idx:
                    V[idx, col] += V00[ri, s]
                for idx in B11_idx:
                    V[idx, col] -= V00[ri, s]
        for ri in range(4):
            if W00[ri, s] != 0:
                W[C01_idx[ri], col] += W00[ri, s]
                W[C11_idx[ri], col] += W00[ri, s]
        col += 1
    
    # P4 = A11(B10-B00)
    for s in range(7):
        for ri in range(4):
            if U00[ri, s] != 0:
                for idx in A11_idx:
                    U[idx, col] += U00[ri, s]
            if V00[ri, s] != 0:
                for idx in B10_idx:
                    V[idx, col] += V00[ri, s]
                for idx in B00_idx:
                    V[idx, col] -= V00[ri, s]
        for ri in range(4):
            if W00[ri, s] != 0:
                W[C00_idx[ri], col] += W00[ri, s]
                W[C10_idx[ri], col] += W00[ri, s]
        col += 1
    
    # P5 = (A00+A01)B11
    for s in range(7):
        for ri in range(4):
            if U00[ri, s] != 0:
                for idx in A00_idx:
                    U[idx, col] += U00[ri, s]
                for idx in A01_idx:
                    U[idx, col] += U00[ri, s]
            if V00[ri, s] != 0:
                for idx in B11_idx:
                    V[idx, col] += V00[ri, s]
        for ri in range(4):
            if W00[ri, s] != 0:
                W[C00_idx[ri], col] -= W00[ri, s]
                W[C01_idx[ri], col] += W00[ri, s]
        col += 1
    
    # P6 = (A10-A00)(B00+B01)
    for s in range(7):
        for ri in range(4):
            if U00[ri, s] != 0:
                for idx in A10_idx:
                    U[idx, col] += U00[ri, s]
                for idx in A00_idx:
                    U[idx, col] -= U00[ri, s]
            if V00[ri, s] != 0:
                for idx in B00_idx:
                    V[idx, col] += V00[ri, s]
                for idx in B01_idx:
                    V[idx, col] += V00[ri, s]
        for ri in range(4):
            if W00[ri, s] != 0:
                W[C11_idx[ri], col] += W00[ri, s]
        col += 1
    
    # P7 = (A01-A11)(B10+B11)
    for s in range(7):
        for ri in range(4):
            if U00[ri, s] != 0:
                for idx in A01_idx:
                    U[idx, col] += U00[ri, s]
                for idx in A11_idx:
                    U[idx, col] -= U00[ri, s]
            if V00[ri, s] != 0:
                for idx in B10_idx:
                    V[idx, col] += V00[ri, s]
                for idx in B11_idx:
                    V[idx, col] += V00[ri, s]
        for ri in range(4):
            if W00[ri, s] != 0:
                W[C00_idx[ri], col] += W00[ri, s]
        col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = U.shape[1]
    # Check shapes
    assert U.shape == (n * m, rank), f"U shape {U.shape} != ({n*m}, {rank})"
    assert V.shape == (m * p, rank), f"V shape {V.shape} != ({m*p}, {rank})"
    assert W.shape == (p * n, rank), f"W shape {W.shape} != ({p*n}, {rank})"
    
    # Random test
    rng = np.random.RandomState(42)
    for trial in range(10):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute via decomposition
        a_flat = A.flatten()  # row-major: [a00, a01, ..., a0m-1, a10, ...]
        b_flat = B.flatten()  # row-major
        u_a = U @ a_flat  # (rank,)
        v_b = V @ b_flat  # (rank,)
        products = u_a * v_b  # (rank,)
        c_flat = W.T @ products  # (p*n,)
        
        # W[k*n+i, r] convention: output index = k*n + i
        # C is n x p, so C[i, k] = c_flat[k*n + i]
        C_computed = c_flat.reshape(p, n).T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            print(f"Verification failed for trial {trial}")
            print(f"Max error: {np.max(np.abs(C_computed - C_expected))}")
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Split 4x5 by 5x6 into sub-problems
    # A (4x5) = [A1 (4x4) | A2 (4x1)]
    # B (5x6) = [B1 (4x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x4 by 4x6 -> split B1 into [B11 (4x4) | B12 (4x2)]
    #   A1*B11: 4x4 by 4x4 (Strassen, rank 49)
    #   A1*B12: 4x4 by 4x2 (schoolbook, rank 32)
    # A2*B2: 4x1 by 1x6 (schoolbook, rank 24)
    # Total: 49 + 32 + 24 = 105
    
    # But we can do better. Let's use Strassen for 4x4 by 4x2 as well.
    # Actually, 4x4 by 4x2 is a special case. Let's just use schoolbook for simplicity
    # and see if we can improve.
    
    # For now, let's try the basic split:
    # A1*B11: Strassen 4x4 by 4x4, rank 49
    # A1*B12: schoolbook 4x4 by 4x2, rank 32
    # A2*B2: schoolbook 4x1 by 1x6, rank 24
    # Total: 105
    
    # Alternative: use Strassen for 4x4 by 4x2
    # 4x4 by 4x2 can be viewed as 4x4 by 4x4 where the second 2 columns of B are zero.
    # This doesn't directly help with rank reduction.
    
    # Let's just go with rank 105 for now and see if it verifies.
    
    if time.time() > deadline:
        # Fallback to schoolbook
        U, V, W = _schoolbook_block(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build Strassen 4x4 by 4x4
    U_s, V_s, W_s = _strassen_4x4()
    
    # Build schoolbook 4x4 by 4x2
    U_s12, V_s12, W_s12 = _schoolbook_block(4, 4, 2)
    
    # Build schoolbook 4x1 by 1x6
    U_s2, V_s2, W_s2 = _schoolbook_block(4, 1, 6)
    
    rank = U_s.shape[1] + U_s12.shape[1] + U_s2.shape[1]
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # A1*B11: A1 is rows 0-3 of A (4x4), B11 is cols 0-3 of B1 (4x4)
    # A1 indices in A (4x5): rows 0-3, cols 0-3 -> flat indices 0-15
    # B11 indices in B (5x6): rows 0-3, cols 0-3 -> flat indices 0-15
    # C output: rows 0-3, cols 0-3 -> C[i,k] at k*4+i
    
    # U_s has shape (16, 49), corresponding to A1 (4x4) flattened
    # V_s has shape (16, 49), corresponding to B11 (4x4) flattened
    # W_s has shape (16, 49), corresponding to C block (4x4) flattened
    
    # A1 flat indices in A (4x5): [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
    # But A is 4x5, so flat index for (i,j) is i*5+j
    # A1 (4x4) elements: A[i,j] for i in 0-3, j in 0-3
    # Flat indices: i*5+j for i in 0-3, j in 0-3
    # = [0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18]
    
    A1_flat = [i*5+j for i in range(4) for j in range(4)]
    # B11 (4x4) elements: B[i,j] for i in 0-3, j in 0-3
    # B is 5x6, flat index for (i,j) is i*6+j
    # = [0,1,2,3, 6,7,8,9, 12,13,14,15, 18,19,20,21]
    B11_flat = [i*6+j for i in range(4) for j in range(4)]
    # C output for A1*B11: C[i,k] for i in 0-3, k in 0-3
    # C is 4x6, flat index (in W convention k*4+i): k*4+i
    # = [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
    C11_flat = [k*4+i for i in range(4) for k in range(4)]
    
    for r in range(49):
        for a_idx, val in zip(A1_flat, U_s[:, r]):
            if val != 0:
                U[a_idx, col + r] = val
        for b_idx, val in zip(B11_flat, V_s[:, r]):
            if val != 0:
                V[b_idx, col + r] = val
        for c_idx, val in zip(C11_flat, W_s[:, r]):
            if val != 0:
                W[c_idx, col + r] = val
    col += 49
    
    # A1*B12: A1 (4x4), B12 (4x2)
    # B12 elements: B[i,j] for i in 0-3, j in 4-5
    # Flat indices: i*6+j for i in 0-3, j in 4-5
    # = [4,5, 10,11, 16,17, 22,23]
    B12_flat = [i*6+j for i in range(4) for j in range(4, 6)]
    # C output for A1*B12: C[i,k] for i in 0-3, k in 4-5
    # W convention: k*4+i
    # = [4*4+0, 4*4+1, 4*4+2, 4*4+3, 5*4+0, 5*4+1, 5*4+2, 5*4+3]
    # = [16,17,18,19, 20,21,22,23]
    C12_flat = [k*4+i for i in range(4) for k in range(4, 6)]
    
    for r in range(32):
        for a_idx, val in zip(A1_flat, U_s12[:, r]):
            if val != 0:
                U[a_idx, col + r] = val
        for b_idx, val in zip(B12_flat, V_s12[:, r]):
            if val != 0:
                V[b_idx, col + r] = val
        for c_idx, val in zip(C12_flat, W_s12[:, r]):
            if val != 0:
                W[c_idx, col + r] = val
    col += 32
    
    # A2*B2: A2 (4x1), B2 (1x6)
    # A2 elements: A[i,4] for i in 0-3
    # Flat indices: i*5+4 = [4, 9, 14, 19]
    A2_flat = [i*5+4 for i in range(4)]
    # B2 elements: B[4,j] for j in 0-5
    # Flat indices: 4*6+j = [24, 25, 26, 27, 28, 29]
    B2_flat = [4*6+j for j in range(6)]
    # C output for A2*B2: C[i,k] for all i in 0-3, k in 0-5
    # W convention: k*4+i
    # = [k*4+i for k in 0-5, i in 0-3]
    C2_flat = [k*4+i for i in range(4) for k in range(6)]
    
    for r in range(24):
        for a_idx, val in zip(A2_flat, U_s2[:, r]):
            if val != 0:
                U[a_idx, col + r] = val
        for b_idx, val in zip(B2_flat, V_s2[:, r]):
            if val != 0:
                V[b_idx, col + r] = val
        for c_idx, val in zip(C2_flat, W_s2[:, r]):
            if val != 0:
                W[c_idx, col + r] = val
    col += 24
    
    # Verify
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p):
            print("Verification failed, falling back to schoolbook")
            U, V, W = _schoolbook_block(n, m, p)
    
    if time.time() > deadline:
        U, V, W = _schoolbook_block(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
