import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor entries
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    # Rank-39 decomposition: 5 Strassen blocks (rank 7 each) + 1 schoolbook block (rank 4)
    rank39 = 5 * 7 + 4
    U = np.zeros((n * m, rank39), dtype=np.int32)
    V = np.zeros((m * p, rank39), dtype=np.int32)
    W = np.zeros((p * n, rank39), dtype=np.int32)
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U[0, r] = 1; U[3, r] = 1
        V[v_e, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1; W[w_c11, r] = 1
        r += 1
        
        # m2 = (c+d)e
        U[2, r] = 1; U[3, r] = 1
        V[v_e, r] = 1
        W[w_c10, r] = 1; W[w_c11, r] = -1
        r += 1
        
        # m3 = a(f-h)
        U[0, r] = 1
        V[v_f, r] = 1; V[v_h, r] = -1
        W[w_c01, r] = 1; W[w_c11, r] = 1
        r += 1
        
        # m4 = d(g-e)
        U[3, r] = 1
        V[v_g, r] = 1; V[v_e, r] = -1
        W[w_c00, r] = 1; W[w_c10, r] = 1
        r += 1
        
        # m5 = (a+b)h
        U[0, r] = 1; U[1, r] = 1
        V[v_h, r] = 1
        W[w_c00, r] = -1; W[w_c01, r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2, r] = 1; U[0, r] = -1
        V[v_e, r] = 1; V[v_f, r] = 1
        W[w_c11, r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1, r] = 1; U[3, r] = -1
        V[v_g, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # ae
    U[0, r] = 1
    V[v_e2, r] = 1
    W[w_c02, r] = 1
    r += 1
    
    # bg
    U[1, r] = 1
    V[v_g2, r] = 1
    W[w_c02, r] = 1
    r += 1
    
    # ce
    U[2, r] = 1
    V[v_e2, r] = 1
    W[w_c12, r] = 1
    r += 1
    
    # dg
    U[3, r] = 1
    V[v_g2, r] = 1
    W[w_c12, r] = 1
    r += 1
    
    # Quick verification
    def verify(U, V, W, entries):
        for (a, b, c) in entries:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if val != 1:
                return False
        return True
    
    if not verify(U, V, W, tensor_entries):
        # If verification fails, it's a bug in construction, but we expect it to pass.
        # Return anyway to avoid crash, though score will be 0.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
