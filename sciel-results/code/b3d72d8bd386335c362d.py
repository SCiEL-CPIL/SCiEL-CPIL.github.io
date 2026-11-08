"""Matrix multiplication 4,4,4 - Strassen rank-49 decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    Uses Strassen's algorithm for 4x4x4 multiplication, achieving rank 49.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Define the 2x2 Strassen decomposition coefficients
    # U2, V2, W2 are 4x7 matrices where rows correspond to input/output elements (0..3)
    # and columns correspond to the 7 Strassen products.
    # Indices for 2x2 matrix: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0], # A00
        [0,  0,  0,  0,  1,  0,  1], # A01
        [0,  1,  0,  0,  0,  1,  0], # A10
        [1,  1,  0,  1,  0,  0, -1]  # A11
    ], dtype=float)
    
    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0], # B00
        [0,  0,  1,  0,  0,  1,  0], # B01
        [0,  0,  0,  1,  0,  0,  1], # B10
        [1,  0, -1,  0,  1,  0,  1]  # B11
    ], dtype=float)
    
    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1], # C00
        [0,  0,  1,  0,  1,  0,  0], # C01
        [0,  1,  0,  1,  0,  0,  0], # C10
        [1, -1, 1,  0,  0,  1,  0]   # C11
    ], dtype=float)
    
    # Verify 2x2 Strassen correctness
    # C = A @ B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: sum_r U2[i, r] V2[j, r] W2[k, r] ? No, the tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # For 2x2: T_{i,j,k} is 1 if k = i? No.
    # Matrix mult tensor: T_{i,j,k} such that C_{ik} = sum_j A_{ij} B_{jk}.
    # The rank-1 tensor is u_i v_j w_k.
    # sum_r U_{i,r} V_{j,r} W_{k,r} should equal 1 if the term A_{ij}B_{jk} contributes to C_{ik}.
    # Actually, the standard definition is that the multiplication tensor T has entries T_{i,j,k} = 1 if k is the output index?
    # Let's check the convention in the prompt:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition is C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Wait, the sum is over r. The indices i,j,k are fixed for a specific output C[i,k] and input pair A[i,j], B[j,k].
    # So for a fixed i, j, k, the value of the tensor T_{i,j,k} is 1 (since A[i,j]*B[j,k] is exactly the term).
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    
    # Let's verify 2x2 with this logic.
    # U2[i, r], V2[j, r], W2[k, r].
    # We need sum_r U2[i,r] V2[j,r] W2[k,r] == 1.
    
    # Let's do a quick check for i=0, j=0, k=0.
    # U2[0] = [1, 0, 1, 0, 1, -1, 0]
    # V2[0] = [1, 1, 0, -1, 0, 1, 0]
    # W2[0] = [1, 0, 0, 1, -1, 0, 1]
    # Products:
    # r=0: 1*1*1 = 1
    # r=1: 0*1*0 = 0
    # r=2: 1*0*0 = 0
    # r=3: 0*-1*1 = 0
    # r=4: 1*0*-1 = 0
    # r=5: -1*1*0 = 0
    # r=6: 0*0*1 = 0
    # Sum = 1. Correct.
    
    # Now construct 4x4x4 decomposition.
    # The 4x4 matrices are split into 2x2 blocks.
    # A[i,j] where i,j in 0..3.
    # Block index for A: bi = i//2, bj = j//2. Local index: li = i%2, lj = j%2.
    # The "super" Strassen factor is indexed by s (0..6) corresponding to the block level.
    # The "sub" Strassen factor is indexed by t (0..6) corresponding to the element level within the block.
    # Total rank = 7 * 7 = 49.
    # Column index in 49-dim space: col = s * 7 + t.
    
    # U[i*m+j, col] = U2_block[i_block, s] * U2_local[i_local, t]
    # Note: U2_block uses the block indices (0..3) as rows?
    # In the 2x2 Strassen derivation, U2 is 4x7. The rows correspond to the 4 elements of the 2x2 matrix.
    # When we apply this to the block level, the "elements" are the 4 blocks of the 4x4 matrix.
    # So U2_block is actually just U2 itself, but interpreted as acting on the 4 blocks.
    # Similarly for V2 and W2.
    
    # So:
    # U[i*4+j, s*7+t] = U2[ (i//2)*2 + (j//2), s ] * U2[ (i%2)*2 + (j%2), t ]
    # V[j*4+k, s*7+t] = V2[ (j//2)*2 + (k//2), s ] * V2[ (j%2)*2 + (k%2), t ]
    # W[k*4+i, s*7+t] = W2[ (i//2)*2 + (k//2), s ] * W2[ (i%2)*2 + (k%2), t ]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Precompute block and local indices
    # For A: index = i*4+j
    # i, j in 0..3
    # block_idx = (i//2)*2 + (j//2)
    # local_idx = (i%2)*2 + (j%2)
    
    for i in range(4):
        for j in range(4):
            idx_a = i * 4 + j
            block_a = (i // 2) * 2 + (j // 2)
            local_a = (i % 2) * 2 + (j % 2)
            
            for s in range(7):
                u_block_val = U2[block_a, s]
                if u_block_val == 0:
                    continue
                for t in range(7):
                    u_local_val = U2[local_a, t]
                    if u_local_val == 0:
                        continue
                    U[idx_a, s * 7 + t] = u_block_val * u_local_val

    for j in range(4):
        for k in range(4):
            idx_b = j * 4 + k
            block_b = (j // 2) * 2 + (k // 2)
            local_b = (j % 2) * 2 + (k % 2)
            
            for s in range(7):
                v_block_val = V2[block_b, s]
                if v_block_val == 0:
                    continue
                for t in range(7):
                    v_local_val = V2[local_b, t]
                    if v_local_val == 0:
                        continue
                    V[idx_b, s * 7 + t] = v_block_val * v_local_val

    for i in range(4):
        for k in range(4):
            # W is indexed by [k*n + i, r] = [4*k + i, r]
            idx_w = k * 4 + i
            block_c = (i // 2) * 2 + (k // 2)
            local_c = (i % 2) * 2 + (k % 2)
            
            for s in range(7):
                w_block_val = W2[block_c, s]
                if w_block_val == 0:
                    continue
                for t in range(7):
                    w_local_val = W2[local_c, t]
                    if w_local_val == 0:
                        continue
                    W[idx_w, s * 7 + t] = w_block_val * w_local_val

    # Verify the decomposition
    # C[i,k] = sum_j sum_r U[i*4+j, r] * V[j*4+k, r] * W[4*k+i, r]
    # We can check this using tensor contraction.
    # Let's pick a few random integer matrices to verify.
    
    np.random.seed(42)
    valid = True
    for _ in range(5):
        A_test = np.random.randint(-5, 5, (4, 4)).astype(float)
        B_test = np.random.randint(-5, 5, (4, 4)).astype(float)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((4, 4))
        for i in range(4):
            for k in range(4):
                total = 0.0
                for j in range(4):
                    u_row = U[i*4+j, :]
                    v_row = V[j*4+k, :]
                    w_row = W[4*k+i, :]
                    total += np.dot(u_row * v_row, w_row)
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            valid = False
            break
    
    if not valid:
        # Fallback to schoolbook rank 64 if verification fails (should not happen)
        rank = 64
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j, col] = 1
                    V[j * 4 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
    else:
        # Convert to lists for output
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
