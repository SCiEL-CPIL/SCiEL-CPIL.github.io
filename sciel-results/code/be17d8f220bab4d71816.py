import numpy as np

def construct():
    n, m, p = 2, 2, 10
    rank = 35
    
    # Strassen vectors for 2x2 multiplication
    # Order of A indices: [a, b, c, d] corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    # Order of B indices: [e, f, g, h] corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
    # Order of C indices: [C11, C21, C12, C22] corresponding to C[0,0], C[1,0], C[0,1], C[1,1]
    
    u_vecs = [
        np.array([1, 0, 0, 1]),
        np.array([0, 0, 1, 1]),
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 0, 1]),
        np.array([1, 1, 0, 0]),
        np.array([-1, 0, 1, 0]),
        np.array([0, 1, 0, -1])
    ]
    
    v_vecs = [
        np.array([1, 0, 0, 1]),
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, -1]),
        np.array([-1, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
        np.array([1, 1, 0, 0]),
        np.array([0, 0, 1, 1])
    ]
    
    w_vecs = [
        np.array([1, 0, 0, 1]),
        np.array([0, 1, 0, -1]),
        np.array([0, 0, 1, 1]),
        np.array([1, 1, 0, 0]),
        np.array([-1, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
        np.array([1, 0, 0, 0])
    ]
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # There are 5 blocks of 2x2 in the 2x10 matrix B
    # B is 2x10. We split it into 5 blocks of 2x2.
    # Block r corresponds to columns 2r and 2r+1 of B.
    # The indices in V for block r are:
    # B[0, 2r] -> index 2r
    # B[0, 2r+1] -> index 2r+1
    # B[1, 2r] -> index 10 + 2r
    # B[1, 2r+1] -> index 10 + 2r + 1
    
    # The indices in W for block r are:
    # C[0, 2r] -> index 4r (C11)
    # C[1, 2r] -> index 4r+1 (C21)
    # C[0, 2r+1] -> index 4r+2 (C12)
    # C[1, 2r+1] -> index 4r+3 (C22)
    
    col = 0
    for r in range(5):
        for k in range(7):
            # U is the same for all blocks
            U[:, col] = u_vecs[k]
            
            # V depends on the block r
            # v_vecs[k] is [e, f, g, h]
            # Map to V indices
            V[2*r, col] = v_vecs[k][0]
            V[2*r+1, col] = v_vecs[k][1]
            V[10 + 2*r, col] = v_vecs[k][2]
            V[10 + 2*r + 1, col] = v_vecs[k][3]
            
            # W depends on the block r
            # w_vecs[k] is [C11, C21, C12, C22]
            # Map to W indices
            W[4*r, col] = w_vecs[k][0]
            W[4*r+1, col] = w_vecs[k][1]
            W[4*r+2, col] = w_vecs[k][2]
            W[4*r+3, col] = w_vecs[k][3]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
