import time
import numpy as np

N, M, P = 2, 3, 10

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Rank 55 decomposition
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x10) into B1 (2x10) and B2 (1x10)
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 (2x2) * B1 (2x10)
    # Split B1 into 5 blocks of 2x2: B1_0 ... B1_4
    # Use Strassen (rank 7) for each block
    # Total rank for Part 1: 5 * 7 = 35
    
    # Part 2: A2 (2x1) * B2 (1x10)
    # Schoolbook: 2 * 1 * 10 = 20
    # Total rank for Part 2: 20
    
    # Total Rank: 55
    
    R = 55
    U = np.zeros((N*M, R))
    V = np.zeros((M*P, R))
    W = np.zeros((P*N, R))
    
    # Indices:
    # U: (i, j) -> i*M + j
    # V: (j, k) -> j*P + k
    # W: (k, i) -> k*N + i
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # We need to map these to U, V, W entries.
    # For a specific block b (0..4), and Strassen term s (0..6):
    # The "scalar" multiplication is between a linear combo of A1 entries and a linear combo of B1 entries.
    
    # A1 indices: i in {0,1}, j in {0,1}.
    # U row index: i*M + j.
    # A1[0,0] -> U[0, t]
    # A1[0,1] -> U[1, t]
    # A1[1,0] -> U[3, t]
    # A1[1,1] -> U[4, t]
    
    # B1 indices: j in {0,1}, k in {2b, 2b+1}.
    # V row index: j*P + k.
    # B1[0, 2b] -> V[0*P + 2b, t]
    # B1[0, 2b+1] -> V[0*P + 2b+1, t]
    # B1[1, 2b] -> V[1*P + 2b, t]
    # B1[1, 2b+1] -> V[1*P + 2b+1, t]
    
    # W indices: k in {2b, 2b+1}, i in {0,1}.
    # W row index: k*N + i.
    # W[2b, 0] -> W[2b*2 + 0, t]
    # W[2b+1, 0] -> W[(2b+1)*2 + 0, t]
    # W[2b, 1] -> W[2b*2 + 1, t]
    # W[2b+1, 1] -> W[(2b+1)*2 + 1, t]
    
    # Strassen terms definition:
    # term: (U_coefs, V_coefs, W_coefs)
    # U_coefs: dict { (i,j): val }
    # V_coefs: dict { (j,k_local): val }
    # W_coefs: dict { (k_local, i): val }
    
    strassen_terms = [
        # m1 = (a+d)(e+h)
        ( {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):-1, (1,0):-1, (1,1):1} ),
        # m2 = (c+d)e
        ( {(1,0):1, (1,1):1}, {(0,0):1}, {(0,0):1, (1,0):1} ),
        # m3 = a(f-h)
        ( {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,0):1} ),
        # m4 = d(g-e)
        ( {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,1):1} ),
        # m5 = (a+b)h
        ( {(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):1} ),
        # m6 = (c-a)(e+f)
        ( {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1} ),
        # m7 = (b-d)(g+h)
        ( {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,0):-1} ),
    ]
    
    t = 0
    # Part 1: 5 blocks of Strassen
    for b in range(5):
        for term in strassen_terms:
            U_coefs, V_coefs, W_coefs = term
            for (i, j), val in U_coefs.items():
                U[i*M + j, t] = val
            for (j, k_local), val in V_coefs.items():
                k_global = 2*b + k_local
                V[j*P + k_global, t] = val
            for (k_local, i), val in W_coefs.items():
                k_global = 2*b + k_local
                W[k_global*N + i, t] = val
            t += 1
            
    # Part 2: A2 (2x1) * B2 (1x10)
    # A2 is column j=2 of A.
    # B2 is row j=2 of B.
    # C2[i, k] = A[ i, 2 ] * B[ 2, k ]
    # For each k in 0..9:
    #   Term: A[0,2] * B[2,k] -> C[0,k]
    #   Term: A[1,2] * B[2,k] -> C[1,k]
    # This is 2 terms per k, total 20 terms.
    
    for k in range(P):
        # Term 1: A[0,2] * B[2,k]
        U[0*M + 2, t] = 1.0
        V[2*P + k, t] = 1.0
        W[k*N + 0, t] = 1.0
        t += 1
        
        # Term 2: A[1,2] * B[2,k]
        U[1*M + 2, t] = 1.0
        V[2*P + k, t] = 1.0
        W[k*N + 1, t] = 1.0
        t += 1
        
    # t should be 55
    assert t == R, f"Expected rank {R}, got {t}"
    
    # Verify
    T = np.zeros((N*M, M*P, P*N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i*M+j, j*P+k, k*N+i] = 1.0
                
    T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback to schoolbook if verification fails (should not happen)
        R_init = 60
        U = np.zeros((N*M, R_init))
        V = np.zeros((M*P, R_init))
        W = np.zeros((P*N, R_init))
        t = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i*M + j, t] = 1.0
                    V[j*P + k, t] = 1.0
                    W[k*N + i, t] = 1.0
                    t += 1
        R = R_init

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
