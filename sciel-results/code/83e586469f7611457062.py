import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 recursive Strassen decomposition for 4x4 matrix multiplication.
    Vectorized construction and verification for speed.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U index: i*4 + k  (A row i, A col k)
      V index: k*4 + j  (B row k, B col j)
      W index: j*4 + i  (C col j, C row i)  [trace convention]
    """
    t0 = time.time()
    deadline = 140.0

    n = m = p = 4
    rank = 49

    # Define 2x2 Strassen terms as 2x2 coefficient matrices
    # Each term: (A_matrix, B_matrix, C_matrix)
    # A_matrix[r][c] = coefficient of A[r,c] in this term
    # Same for B and C
    
    # Using numpy arrays for speed
    s2 = [
        # t0: (A00+A11)*(B00+B11)
        (np.array([[1,0],[0,1]], dtype=np.int8),
         np.array([[1,0],[0,1]], dtype=np.int8),
         np.array([[1,0],[0,1]], dtype=np.int8)),
        # t1: (A10+A11)*B00
        (np.array([[0,0],[1,1]], dtype=np.int8),
         np.array([[1,0],[0,0]], dtype=np.int8),
         np.array([[0,0],[1,-1]], dtype=np.int8)),
        # t2: A00*(B01-B11)
        (np.array([[1,0],[0,0]], dtype=np.int8),
         np.array([[0,1],[0,-1]], dtype=np.int8),
         np.array([[0,1],[0,1]], dtype=np.int8)),
        # t3: A11*(B10-B00)
        (np.array([[0,0],[0,1]], dtype=np.int8),
         np.array([[-1,0],[1,0]], dtype=np.int8),
         np.array([[1,0],[1,0]], dtype=np.int8)),
        # t4: (A00+A01)*B11
        (np.array([[1,1],[0,0]], dtype=np.int8),
         np.array([[0,0],[0,1]], dtype=np.int8),
         np.array([[-1,1],[0,0]], dtype=np.int8)),
        # t5: (A10-A00)*(B00+B01)
        (np.array([[-1,0],[1,0]], dtype=np.int8),
         np.array([[1,1],[0,0]], dtype=np.int8),
         np.array([[0,0],[0,1]], dtype=np.int8)),
        # t6: (A01-A11)*(B10+B11)
        (np.array([[0,1],[0,-1]], dtype=np.int8),
         np.array([[0,0],[1,1]], dtype=np.int8),
         np.array([[1,0],[0,0]], dtype=np.int8)),
    ]

    # Precompute Kronecker products for outer*inner coefficients
    # For each outer term t and inner term k, the coefficient for A[i,ak] is:
    #   outer_A[i//2, ak//2] * inner_A[i%2, ak%2]
    # This is exactly the Kronecker product outer_A ⊗ inner_A (with appropriate ordering)
    
    # Kronecker product: (A⊗B)[i*2+r, j*2+c] = A[i,j]*B[r,c]
    # Our indexing: global (i, ak) -> block (i//2, ak//2), local (i%2, ak%2)
    # So U[i*4+ak] corresponds to (A⊗B)[i*2+local_r, ak*2+local_c] where i=block_r*2+local_r
    # Actually: global index i = block_r * 2 + local_r, so i*2 + local_r = (block_r*2+local_r)*2 + local_r
    # That's not standard Kronecker. Let me think again.
    
    # Standard Kronecker: (A⊗B) is 4x4 where (A⊗B)[i*2+r, j*2+c] = A[i,j]*B[r,c]
    # Our A global index: i (0..3), ak (0..3)
    # i = block_r * 2 + local_r, ak = block_c * 2 + local_c
    # So (A⊗B)[i*2+local_r, ak*2+local_c] doesn't match our indexing directly.
    
    # Let me just use the Kronecker product and reshape appropriately.
    # (outer_A ⊗ inner_A) has shape (4,4) with:
    #   [i*2+r, j*2+c] = outer_A[i,j] * inner_A[r,c]
    # We want U[i*4+ak] = outer_A[i//2, ak//2] * inner_A[i%2, ak%2]
    # Let i = 2*i_block + i_local, ak = 2*ak_block + ak_local
    # Then (outer_A ⊗ inner_A)[i_block*2 + i_local, ak_block*2 + ak_local] = outer_A[i_block, ak_block] * inner_A[i_local, ak_local]
    # But that's index (i_block*2+i_local, ak_block*2+ak_local) in the 4x4 matrix
    # Our U index is i*4+ak = (2*i_block+i_local)*4 + (2*ak_block+ak_local)
    # The Kronecker matrix index is (i_block*2+i_local)*4 + (ak_block*2+ak_local)
    # These are the same! Because row = i_block*2+i_local = i, col = ak_block*2+ak_local = ak
    # So U_col = (outer_A ⊗ inner_A).flatten() where we flatten row-major
    
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)

    for t in range(7):
        oA, oB, oC = s2[t]
        for k in range(7):
            iA, iB, iC = s2[k]
            idx = t * 7 + k
            
            # Kronecker product gives 4x4 matrix
            # Flatten row-major: index = row*4 + col
            U[:, idx] = np.kron(oA, iA).flatten().astype(np.float64)
            V[:, idx] = np.kron(oB, iB).flatten().astype(np.float64)
            # W uses trace convention: W[j*4+i] = C[i,j]
            # So we need to transpose the C matrix before flattening
            # C[i,j] -> W index j*4+i
            # If C_kron = oC ⊗ iC, then C_kron[i,j] = oC[i//2,j//2]*iC[i%2,j%2]
            # W[j*4+i] = C_kron[i,j]
            # So W = C_kron.T.flatten()
            W[:, idx] = np.kron(oC, iC).T.flatten().astype(np.float64)

    # Vectorized verification
    # For each (i,k,j): sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r] should equal 1
    # Build tensors: U has shape (16, 49), V (16, 49), W (16, 49)
    # Reshape U to (4, 4, 49) where U[i,k,r]
    # Reshape V to (4, 4, 49) where V[k,j,r]
    # Reshape W to (4, 4, 49) where W[j,i,r] (since W index is j*4+i)
    
    U3 = U.reshape(4, 4, 49)  # U3[i, k, r]
    V3 = V.reshape(4, 4, 49)  # V3[k, j, r]
    W3 = W.reshape(4, 4, 49)  # W3[j, i, r]
    
    # Compute sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] for all i,k,j
    # Use einsum: result[i,k,j] = sum_r U3[i,k,r] * V3[k,j,r] * W3[j,i,r]
    result = np.einsum('ikr,kjr,jir->ikj', U3, V3, W3)
    
    max_err = np.max(np.abs(result - 1.0))
    
    if max_err > 1e-8:
        # Fallback to schoolbook rank-64
        U64 = np.zeros((16, 64), dtype=np.float64)
        V64 = np.zeros((16, 64), dtype=np.float64)
        W64 = np.zeros((16, 64), dtype=np.float64)
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U64[i*4+k, idx] = 1.0
                    V64[k*4+j, idx] = 1.0
                    W64[j*4+i, idx] = 1.0
                    idx += 1
        return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}

    # Check deadline
    if time.time() > deadline:
        # Return what we have
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
