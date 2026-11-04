import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays with shape (4, 7).
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 4
    start_time = time.time()
    deadline = start_time + 140
    
    U2, V2, W2 = _strassen_2x2()
    
    # Reshape to (2, 2, 7)
    # U2 indices: (i, k) flattened to 4. We want (i1, i2, k1, k2, r1, r2)
    # U2[i, k] where i=2*i1+i2, k=2*k1+k2? No, U2 is 2x2 blocks.
    # In 2x2, i is 0..1, k is 0..1.
    # So U2[i, k] corresponds to i1=i, k1=k.
    # We need to map 4x4 indices.
    
    # U2_reshaped[i1, k1, r1] = U2[2*i1 + k1, r1]?
    # Wait, in _strassen_2x2, U is 4x7.
    # Index 0: i=0, k=0?
    # Let's check the mapping in _strassen_2x2.
    # U[0, 0] = 1. This is A00. i=0, k=0.
    # U[3, 0] = 1. This is A11. i=1, k=1.
    # So row index of U2 is 2*i + k?
    # If i=0, k=0 -> 0.
    # If i=1, k=1 -> 3.
    # If i=0, k=1 -> 1.
    # If i=1, k=0 -> 2.
    # Yes, row index = 2*i + k.
    
    # So U2[i, k] is stored at U2[2*i + k, r].
    # To get U2[i1, k1, r1], we access U2[2*i1 + k1, r1].
    
    # We want to form U4[i, k, r] where i=2*i1+i2, k=2*k1+k2, r=7*r1+r2.
    # U4[i, k, r] = U2[2*i1+k1, r1] * U2[2*i2+k2, r2].
    
    # Let's create a 3D tensor U2_t[i1, k1, r1] = U2[2*i1+k1, r1].
    # We can do this by reshaping and permuting.
    # U2 is (4, 7).
    # Reshape to (2, 2, 7) ?
    # If we reshape (4, 7) to (2, 2, 7), the first axis splits 4 into 2,2.
    # Index (a, b, c) corresponds to row a*2+b.
    # We want row 2*i1 + k1. So a=i1, b=k1.
    # So U2_reshaped[i1, k1, r1] = U2.reshape(2, 2, 7)[i1, k1, r1].
    
    U2_t = U2.reshape(2, 2, 7)
    V2_t = V2.reshape(2, 2, 7)
    W2_t = W2.reshape(2, 2, 7)
    
    # Compute tensor product
    # U4_t[i1, i2, k1, k2, r1, r2] = U2_t[i1, k1, r1] * U2_t[i2, k2, r2]
    U4_t = np.einsum('ikr,jls->ijklrs', U2_t, U2_t)
    V4_t = np.einsum('kjl,klm->jklmjl' , V2_t, V2_t) # Wait, V indices are (k, j).
    # V2_t[k1, j1, r1].
    # V4_t[k1, k2, j1, j2, r1, r2] = V2_t[k1, j1, r1] * V2_t[k2, j2, r2].
    # Let's be careful with einsum labels.
    # V2_t axes: k1, j1, r1.
    # We want V4_t axes: k1, k2, j1, j2, r1, r2.
    # V4_t[k1, k2, j1, j2, r1, r2] = V2_t[k1, j1, r1] * V2_t[k2, j2, r2].
    # np.einsum('kjr,klm->kjklmr', V2_t, V2_t) ? No.
    # Let's use explicit broadcasting or simpler einsum.
    # V4_t = V2_t[:, None, :, None, :, None] * V2_t[None, :, None, :, None, :]
    # Axes: k1, k2, j1, j2, r1, r2.
    # V2_t shape (2, 2, 7).
    # V2_t[:, np.newaxis, :, np.newaxis, :, np.newaxis] shape (2, 1, 2, 1, 7, 1).
    # V2_t[np.newaxis, :, np.newaxis, :, np.newaxis, :] shape (1, 2, 1, 2, 1, 7).
    # Product shape (2, 2, 2, 2, 7, 7).
    # Indices: k1, k2, j1, j2, r1, r2.
    
    V4_t = V2_t[:, np.newaxis, :, np.newaxis, :, np.newaxis] * V2_t[np.newaxis, :, np.newaxis, :, np.newaxis, :]
    
    # W4_t[j1, j2, i1, i2, r1, r2] = W2_t[j1, i1, r1] * W2_t[j2, i2, r2].
    # W2_t axes: j1, i1, r1.
    # W4_t axes: j1, j2, i1, i2, r1, r2.
    W4_t = W2_t[:, np.newaxis, :, np.newaxis, :, np.newaxis] * W2_t[np.newaxis, :, np.newaxis, :, np.newaxis, :]
    
    # Now reshape to (4, 4, 49)
    # U4_t: (2, 2, 2, 2, 7, 7) -> (i1, i2, k1, k2, r1, r2)
    # We want (i, k, r) where i=2*i1+i2, k=2*k1+k2, r=7*r1+r2.
    # Transpose to (i1, i2, k1, k2, r1, r2) is already the order?
    # U4_t axes: i1, i2, k1, k2, r1, r2.
    # We need to group (i1, i2) -> i, (k1, k2) -> k, (r1, r2) -> r.
    # This is a specific transpose/reshape.
    # np.reshape will flatten in C-order.
    # We need to ensure the memory layout matches the index calculation.
    # i = 2*i1 + i2.
    # If we have array A[i1, i2], reshape to (4) gives index 2*i1 + i2. Correct.
    # So we just need to group the axes.
    # U4_reshaped = U4_t.reshape(4, 4, 49).
    # Let's verify.
    # U4_t[i1, i2, k1, k2, r1, r2].
    # Reshape to (4, 4, 49).
    # First dim: 2*2=4. Index 2*i1+i2.
    # Second dim: 2*2=4. Index 2*k1+k2.
    # Third dim: 7*7=49. Index 7*r1+r2.
    # This matches exactly.
    
    U4 = U4_t.reshape(4, 4, 49)
    V4 = V4_t.reshape(4, 4, 49)
    W4 = W4_t.reshape(4, 4, 49)
    
    # Flatten to (16, 49)
    # U4[i, k, r] -> U_flat[4*i + k, r].
    # Reshape (4, 4, 49) to (16, 49).
    # Index 4*i + k.
    # C-order flatten of (4, 4) gives 4*i + k. Correct.
    
    U_flat = U4.reshape(16, 49)
    V_flat = V4.reshape(16, 49)
    W_flat = W4.reshape(16, 49)
    
    # Convert to list of lists
    return {"U": U_flat.tolist(), "V": V_flat.tolist(), "W": W_flat.tolist()}
