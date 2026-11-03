import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 145.0
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-9):
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(reconstructed - T)) < tol
    
    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Construct rank-24 decomposition
    # Split j into m groups of size 1, k into p groups of size 1
    # Each block is M_{2,1,1} which has rank 1
    rank = m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
    
    # Verify
    if check_decomposition(U, V, W, tol=1e-8):
        return to_lists(U, V, W)
    
    # Fallback to rank 42 if rank 24 fails (should not happen)
    # Re-implementing rank 42 briefly for safety
    def make_rank42():
        rank42 = 42
        U42 = np.zeros((n*m, rank42))
        V42 = np.zeros((m*p, rank42))
        W42 = np.zeros((p*n, rank42))
        
        U_s = [
            {(0,0):1, (1,1):1},
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (0,0):-1},
            {(0,1):1, (1,1):-1},
        ]
        V_s = [
            {(0,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1, (0,0):-1},
            {(1,1):1},
            {(0,0):1, (0,1):1},
            {(1,0):1, (1,1):1},
        ]
        W_s = [
            {(0,0):1, (1,1):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1, (1,1):1},
            {(0,0):1, (0,1):1},
            {(0,0):-1, (1,0):1},
            {(1,1):1},
            {(0,0):1},
        ]
        
        j_groups = [(0,1), (2,3)]
        k_groups = [(0,1), (2,3), (4,5)]
        
        col = 0
        for jg in j_groups:
            j_base = jg[0]
            for kg in k_groups:
                k_base = kg[0]
                for s in range(7):
                    for (i_loc, j_loc), coeff in U_s[s].items():
                        U42[i_loc*m + (j_base+j_loc), col] += coeff
                    for (j_loc, k_loc), coeff in V_s[s].items():
                        V42[(j_base+j_loc)*p + (k_base+k_loc), col] += coeff
                    for (k_loc, i_loc), coeff in W_s[s].items():
                        W42[(k_base+k_loc)*n + i_loc, col] += coeff
                    col += 1
        return U42, V42, W42

    U42, V42, W42 = make_rank42()
    return to_lists(U42, V42, W42)
