import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 115.0
_MARGIN = 5.0


def _time_ok():
    return time.time() < _DEADLINE - _MARGIN


def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors (rank 7)
    # Using float for calculation, but casting to int8 for output as required by "integer/half-integer"
    # The values are small integers.
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.int8)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.int8)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.int8)

    R = 49
    
    # We need to construct U, V, W of shape (16, 49)
    # The decomposition is the Kronecker product of the 2x2 decompositions.
    # U_4[i, j, a, b] = U2[i_block, a_block] * U2[j_block, b_block] ?
    # Based on the seed's logic:
    # i = i1*2 + i2, j = j1*2 + j2
    # idx1 = i1*2 + j1
    # idx2 = i2*2 + j2
    # U_flat[i*4+j, a*7+b] = U2[idx1, a] * U2[idx2, b]
    
    # Let's vectorize this construction.
    # i1, i2 in {0,1}. j1, j2 in {0,1}.
    # idx1 in {0,1,2,3}. idx2 in {0,1,2,3}.
    # a in {0..6}. b in {0..6}.
    
    # Create indices for i1, i2, j1, j2
    i1s = np.array([0, 0, 1, 1]) # i=0,1 -> i1=0; i=2,3 -> i1=1
    i2s = np.array([0, 1, 0, 1])
    j1s = np.array([0, 0, 0, 0]) # j=0,1 -> j1=0? No.
    # j = 0 (00), 1 (01), 2 (10), 3 (11)
    # j1 = j // 2. j2 = j % 2.
    js = np.array([0, 1, 2, 3])
    j1s = js // 2
    j2s = js % 2
    
    is_ = np.array([0, 1, 2, 3])
    i1s = is_ // 2
    i2s = is_ % 2
    
    # idx1 = i1*2 + j1
    # idx2 = i2*2 + j2
    # We need a grid of (i, j) pairs.
    # Shape of U is (16, 49).
    # Row index r = i*4 + j.
    # Col index c = a*7 + b.
    
    # Let's build the 4D tensor U_4[i, j, a, b] = U2[i1*2+j1, a] * U2[i2*2+j2, b]
    # Then reshape to (16, 49).
    
    if not _time_ok():
        # Fallback to empty if time is critical (shouldn't happen)
        return {"U": np.zeros((16, 49), dtype=np.int8).tolist(), 
                "V": np.zeros((16, 49), dtype=np.int8).tolist(), 
                "W": np.zeros((16, 49), dtype=np.int8).tolist()}

    # Calculate idx1 and idx2 for all i, j
    # i1 shape (4,), j1 shape (4,) -> idx1 shape (4, 4) via broadcasting
    idx1 = i1s[:, None] * 2 + j1s[None, :]
    idx2 = i2s[:, None] * 2 + j2s[None, :]
    
    # U2 is (4, 7).
    # We want U_tensor[i, j, a, b] = U2[idx1[i,j], a] * U2[idx2[i,j], b]
    
    # U2[idx1] has shape (4, 4, 7) -> (i, j, a)
    # U2[idx2] has shape (4, 4, 7) -> (i, j, b)
    
    U_part1 = U2[idx1] # (4, 4, 7)
    U_part2 = U2[idx2] # (4, 4, 7)
    
    # U_tensor = U_part1[:, :, :, None] * U_part2[:, :, None, :]
    # Shape: (4, 4, 7, 7)
    U_tensor = U_part1[:, :, :, None] * U_part2[:, :, None, :]
    
    # Reshape to (16, 49)
    # i, j order: i outer, j inner. So (4,4) -> 16.
    # a, b order: a outer, b inner. So (7,7) -> 49.
    U = U_tensor.reshape(16, 49).astype(np.int8)
    
    if not _time_ok():
        return {"U": U.tolist(), "V": np.zeros((16, 49), dtype=np.int8).tolist(), 
                "W": np.zeros((16, 49), dtype=np.int8).tolist()}

    # V construction:
    # V[j, k] = outer(V2[idx1_v], V2[idx2_v])
    # j = j1*2 + j2? No, in the seed:
    # for j in range(4): j1, j2 = j>>1, j&1
    # for k in range(4): k1, k2 = k>>1, k&1
    # idx1 = j1*2 + k1
    # idx2 = j2*2 + k2
    # V[j*4+k] = outer(V2[idx1], V2[idx2])
    
    ks = np.array([0, 1, 2, 3])
    k1s = ks // 2
    k2s = ks % 2
    
    # V uses j and k
    # idx1_v = j1*2 + k1
    # idx2_v = j2*2 + k2
    idx1_v = j1s[:, None] * 2 + k1s[None, :]
    idx2_v = j2s[:, None] * 2 + k2s[None, :]
    
    V_part1 = V2[idx1_v] # (4, 4, 7)
    V_part2 = V2[idx2_v] # (4, 4, 7)
    
    V_tensor = V_part1[:, :, :, None] * V_part2[:, :, None, :]
    V = V_tensor.reshape(16, 49).astype(np.int8)
    
    if not _time_ok():
        return {"U": U.tolist(), "V": V.tolist(), "W": np.zeros((16, 49), dtype=np.int8).tolist()}

    # W construction:
    # W[k, i] = outer(W2[idx1_w], W2[idx2_w])
    # for k in range(4): k1, k2 = k>>1, k&1
    # for i in range(4): i1, i2 = i>>1, i&1
    # idx1 = k1*2 + i1
    # idx2 = k2*2 + i2
    # W[k*4+i] = outer(W2[idx1], W2[idx2])
    
    idx1_w = k1s[:, None] * 2 + i1s[None, :]
    idx2_w = k2s[:, None] * 2 + i2s[None, :]
    
    W_part1 = W2[idx1_w] # (4, 4, 7)
    W_part2 = W2[idx2_w] # (4, 4, 7)
    
    W_tensor = W_part1[:, :, :, None] * W_part2[:, :, None, :]
    W = W_tensor.reshape(16, 49).astype(np.int8)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
