import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # The tensor is a direct sum of 8 copies of the 2x2x2 multiplication tensor.
    # Rank of 2x2x2 is 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen vectors for 2x2 matrix multiplication
    # Local indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Note: Global W index is k*n + i.
    # For block b, local col k2 maps to global col k = 2*b + k2.
    # Local row i maps to global row i.
    # So global W index for local (i, k2) is (2*b + k2)*n + i.
    # Since n=2, this is 4*b + 2*k2 + i.
    
    # Strassen 7 products
    # u_s: coefficients for A
    # v_s: coefficients for B
    # w_s: coefficients for C
    
    u_vecs = [
        [1, 0, 0, 1],  # M1: a + d
        [0, 0, 1, 1],  # M2: c + d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a + b
        [-1, 0, 1, 0], # M6: c - a
        [0, 1, 0, -1]  # M7: b - d
    ]
    
    v_vecs = [
        [1, 0, 0, 1],  # M1: e + h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f - h
        [-1, 0, 1, 0], # M4: g - e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e + f
        [0, 0, 1, 1]   # M7: g + h
    ]
    
    w_vecs = [
        [1, 0, 0, 1],  # M1: c00 + c11
        [0, 0, 1, -1], # M2: c10 - c11
        [0, 1, 0, 1],  # M3: c01 + c11
        [1, 0, 1, 0],  # M4: c00 + c10
        [-1, 1, 0, 0], # M5: -c00 + c01
        [0, 0, 0, 1],  # M6: c11
        [1, 0, 0, 0]   # M7: c00
    ]

    for b in range(8):
        for s in range(7):
            col = b * 7 + s
            
            # Fill U (shared across blocks, but we write it for each block's column)
            for li in range(4):
                U[li, col] = u_vecs[s][li]
                
            # Fill V
            # Local B index li -> Global V index
            # li=0 -> B[0, 2b] -> V[0*16 + 2b] = V[2b]
            # li=1 -> B[0, 2b+1] -> V[0*16 + 2b+1] = V[2b+1]
            # li=2 -> B[1, 2b] -> V[1*16 + 2b] = V[16+2b]
            # li=3 -> B[1, 2b+1] -> V[1*16 + 2b+1] = V[16+2b+1]
            v_indices = [2*b, 2*b+1, 16+2*b, 16+2*b+1]
            for li in range(4):
                V[v_indices[li], col] = v_vecs[s][li]
                
            # Fill W
            # Local C index li -> Global W index
            # Local C (i, k2) where li = i*2 + k2? 
            # My w_vecs use indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
            # li=0 -> i=0, k2=0 -> Global W[(2b+0)*2 + 0] = W[4b]
            # li=1 -> i=0, k2=1 -> Global W[(2b+1)*2 + 0] = W[4b+2]
            # li=2 -> i=1, k2=0 -> Global W[(2b+0)*2 + 1] = W[4b+1]
            # li=3 -> i=1, k2=1 -> Global W[(2b+1)*2 + 1] = W[4b+3]
            w_indices = [4*b, 4*b+2, 4*b+1, 4*b+3]
            for li in range(4):
                W[w_indices[li], col] = w_vecs[s][li]

    # Verification (optional but good for sanity check, keep it fast)
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        A_test = rng.standard_normal((n, m))
        B_test = rng.standard_normal((m, p))
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            a_part = np.dot(U[:, r], A_test.flatten())
            b_part = np.dot(V[:, r], B_test.flatten())
            prod = a_part * b_part
            C_computed += np.outer(W[:, r], prod).reshape(n, p) # W is (p*n, rank), so W[:,r] is (p*n,). 
            # Wait, W[:,r] is vector of length p*n.
            # C[i,k] += W[k*n+i, r] * prod
            # My loop above: C_computed[i,k] += W[k*n+i, r] * prod
            # The vectorized version:
            # C_flat = np.zeros(n*p)
            # C_flat += W[:, r] * prod
            # C_computed = C_flat.reshape(p, n).T ? 
            # W index is k*n + i. So W_flat[k*n+i] corresponds to C[i,k].
            # So C_flat = W[:,r] * prod.
            # C_computed[i,k] = C_flat[k*n+i].
            # So C_computed = C_flat.reshape(p, n).T
            
        C_computed_correct = np.zeros((n, p))
        for r in range(rank):
            a_part = np.dot(U[:, r], A_test.flatten())
            b_part = np.dot(V[:, r], B_test.flatten())
            prod = a_part * b_part
            W_col = W[:, r]
            # W_col is length p*n. Index k*n + i.
            # C[i,k] += W_col[k*n+i] * prod
            C_computed_correct += (W_col * prod).reshape(p, n).T

        if not np.allclose(C_computed_correct, C_expected, atol=1e-10):
            # Fallback to rank 64 if verification fails (should not happen)
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

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
