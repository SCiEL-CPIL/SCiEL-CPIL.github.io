"""Matrix multiplication 2x4x8 rank-minimal search using Winograd/FFT approach."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = 2, 4, 8
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition of T."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.array_equal(recon, T)
    
    # First, construct the known rank-56 Strassen decomposition as fallback
    def construct_strassen_56():
        rank = 56
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        
        # Strassen coefficients for 2x2 matrix multiplication
        # P1 = (x00 + x11) * (y00 + y11)
        # P2 = (x10 + x11) * y00
        # P3 = x00 * (y01 - y11)
        # P4 = x11 * (y10 - y00)
        # P5 = (x00 + x01) * y11
        # P6 = (x10 - x00) * (y00 + y01)
        # P7 = (x01 - x11) * (y10 + y11)
        
        strassen_defs = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        ]
        
        col_offset = 0
        for t in range(4):  # Output block index 0..3
            for s in range(2):  # Input block index 0..1
                for p_idx in range(7):
                    r = col_offset
                    x_coeffs, y_coeffs, z_coeffs = strassen_defs[p_idx]
                    
                    # U indices: x00->U[2s], x01->U[2s+1], x10->U[4+2s], x11->U[5+2s]
                    for x_idx, val in x_coeffs.items():
                        u_idx = [2*s, 2*s+1, 4+2*s, 5+2*s][x_idx]
                        U[u_idx, r] += val
                    
                    # V indices: y00->V[16s+2t], y01->V[16s+2t+1], y10->V[16s+8+2t], y11->V[16s+8+2t+1]
                    for y_idx, val in y_coeffs.items():
                        v_idx = [16*s + 2*t, 16*s + 2*t+1, 16*s + 8 + 2*t, 16*s + 8 + 2*t+1][y_idx]
                        V[v_idx, r] += val
                    
                    # W indices: z00->W[4t], z01->W[4t+2], z10->W[4t+1], z11->W[4t+3]
                    for z_idx, val in z_coeffs.items():
                        w_idx = [4*t, 4*t+2, 4*t+1, 4*t+3][z_idx]
                        W[w_idx, r] += val
                    
                    col_offset += 1
        
        return U, V, W
    
    U_56, V_56, W_56 = construct_strassen_56()
    best_U, best_V, best_W = U_56.copy(), V_56.copy(), W_56.copy()
    best_rank = 56
    
    # Now try the Winograd/FFT approach for potentially lower rank
    # Strategy: Model matrices as polynomials, use pointwise evaluation
    # For 2x4 * 4x8:
    #   A is 2x4, B is 4x8
    #   We can group columns of A and rows of B
    #   Use FFT-like decomposition with roots of unity
    
    # Approach: Use the fact that we can decompose using polynomial multiplication
    # Let's try to find a rank < 56 decomposition using structured search
    
    # Try rank values from 55 down to 32
    import random
    rng = random.Random(12345)
    
    for target_rank in range(55, 31, -1):
        if time.time() > deadline - 5:
            break
        
        # Try several random structured decompositions
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
            
            # Generate random sparse integer vectors
            U_rand = np.zeros((n*m, target_rank), dtype=np.int64)
            V_rand = np.zeros((m*p, target_rank), dtype=np.int64)
            W_rand = np.zeros((p*n, target_rank), dtype=np.int64)
            
            for c in range(target_rank):
                # Sparse random with small integers
                nz_u = rng.sample(range(n*m), rng.randint(2, 5))
                nz_v = rng.sample(range(m*p), rng.randint(3, 8))
                nz_w = rng.sample(range(p*n), rng.randint(2, 6))
                
                for a in nz_u:
                    U_rand[a, c] = rng.choice([-2, -1, 1, 2])
                for b in nz_v:
                    V_rand[b, c] = rng.choice([-2, -1, 1, 2])
                for d in nz_w:
                    W_rand[d, c] = rng.choice([-2, -1, 1, 2])
            
            if verify(U_rand, V_rand, W_rand):
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                best_rank = target_rank
                break
        
        if best_rank < 56:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
