"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
    Uses block-diagonal Strassen (rank 7 per block, 5 blocks -> rank 35).
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = 35
    
    # Initialize factor matrices
    # U: (n*m, rank) = (4, 35)
    # V: (m*p, rank) = (20, 35)
    # W: (p*n, rank) = (20, 35)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen coefficients for 2x2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # The standard Strassen algorithm for C = A*B (2x2) uses 7 products.
    # Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]].
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e)
    # P3 = (a)(f-h)
    # P4 = (d)(g-e)
    # P5 = (a+b)(h)
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # We map these to the indices in U, V, W.
    # U corresponds to coefficients of A.
    # V corresponds to coefficients of B.
    # W corresponds to coefficients of C.
    
    # C indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) 
    # Note: The task says W[k*n+i, r]. So W index is k*n + i.
    # C[0,0] -> 0*2+0 = 0
    # C[1,0] -> 1*2+0 = 2? No. 
    # "W[k*n+i,r]". 
    # Standard matrix C is 2x10. C[i,k]. 
    # The task says "output factor follows the official trace convention W[k*n+i,r]".
    # Usually C[i,j] is stored as i*p + j or j*n + i.
    # Here W has shape (p*n, rank) = (20, rank).
    # The index is k*n + i. Here k is the column index of C (0..9), i is the row index (0..1).
    # So W[0*2+0] = W[0] corresponds to C[0,0].
    # W[0*2+1] = W[1] corresponds to C[1,0].
    # W[1*2+0] = W[2] corresponds to C[0,1].
    # W[1*2+1] = W[3] corresponds to C[1,1].
    
    # Let's verify the Strassen output formulas for C elements:
    # C00 = P1 - P4 + P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So for each block b (0..4):
    # The B columns are 2b, 2b+1.
    # The C columns are 2b, 2b+1.
    # C[0, 2b] corresponds to W[2b*2 + 0] = W[4b]
    # C[1, 2b] corresponds to W[2b*2 + 1] = W[4b+1]
    # C[0, 2b+1] corresponds to W[(2b+1)*2 + 0] = W[4b+2]
    # C[1, 2b+1] corresponds to W[(2b+1)*2 + 1] = W[4b+3]
    
    # U coefficients (for A):
    # P1: a+d -> [1, 0, 0, 1]
    # P2: c+d -> [0, 0, 1, 1]
    # P3: a   -> [1, 0, 0, 0]
    # P4: d   -> [0, 0, 0, 1]
    # P5: a+b -> [1, 1, 0, 0]
    # P6: c-a -> [-1, 0, 1, 0]
    # P7: b-d -> [0, 1, 0, -1]
    
    u_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V coefficients (for B block [[e, f], [g, h]]):
    # Indices in V: e=2b, f=2b+1, g=10+2b, h=10+2b+1
    # P1: e+h -> [1, 0, 0, 1] (e, f, g, h)
    # P2: e   -> [1, 0, 0, 0]
    # P3: f-h -> [0, 1, 0, -1]
    # P4: g-e -> [0, 0, 1, -1]
    # P5: h   -> [0, 0, 0, 1]
    # P6: e+f -> [1, 1, 0, 0]
    # P7: g+h -> [0, 0, 1, 1]
    
    v_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W coefficients (for C block [[c00, c01], [c10, c11]]):
    # Indices in W: c00=4b, c10=4b+1, c01=4b+2, c11=4b+3
    # C00 = P1 - P4 + P5 - P7
    # C10 = P2 + P4
    # C01 = P3 + P5
    # C11 = P1 - P2 + P3 + P6
    
    # So for each term s (0..6), we add the coefficient to the corresponding W index.
    # Term 0 (P1): C00 += 1, C11 += 1
    # Term 1 (P2): C10 += 1, C11 += -1
    # Term 2 (P3): C01 += 1, C11 += 1
    # Term 3 (P4): C00 += -1, C10 += 1
    # Term 4 (P5): C00 += 1, C01 += 1
    # Term 5 (P6): C11 += 1
    # Term 6 (P7): C00 += -1
    
    w_coeffs = [
        [1, 0, 0, 1],  # P1: C00=1, C10=0, C01=0, C11=1
        [0, 1, 0, -1], # P2: C00=0, C10=1, C01=0, C11=-1
        [0, 0, 1, 1],  # P3: C00=0, C10=0, C01=1, C11=1
        [-1, 1, 0, 0], # P4: C00=-1, C10=1, C01=0, C11=0
        [1, 0, 1, 0],  # P5: C00=1, C10=0, C01=1, C11=0
        [0, 0, 0, 1],  # P6: C00=0, C10=0, C01=0, C11=1
        [-1, 0, 0, 0]  # P7: C00=-1, C10=0, C01=0, C11=0
    ]
    
    for b in range(5):
        for s in range(7):
            col = b * 7 + s
            
            # Fill U
            U[:, col] = u_coeffs[s]
            
            # Fill V
            e = 2 * b
            f = 2 * b + 1
            g = 10 + 2 * b
            h = 10 + 2 * b + 1
            
            V[e, col] = v_coeffs[s][0]
            V[f, col] = v_coeffs[s][1]
            V[g, col] = v_coeffs[s][2]
            V[h, col] = v_coeffs[s][3]
            
            # Fill W
            c00 = 4 * b
            c10 = 4 * b + 1
            c01 = 4 * b + 2
            c11 = 4 * b + 3
            
            W[c00, col] = w_coeffs[s][0]
            W[c10, col] = w_coeffs[s][1]
            W[c01, col] = w_coeffs[s][2]
            W[c11, col] = w_coeffs[s][3]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
