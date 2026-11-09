import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indices based on the problem description
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (official trace convention)
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                
                col += 1
                
    # Verification (optional but good for debugging/correctness)
    # We verify that the decomposition holds for a few random samples or all samples if small enough
    # Here n*m*p = 56, so checking all 56 non-zero entries is fast.
    # We also need to ensure that for non-matching indices, the sum is 0.
    # This is guaranteed by the construction (each column has only 1 non-zero in U, V, W respectively,
    # and they align perfectly for the target tensor entry).
    # Specifically, for a fixed column l, U[:,l] is a unit vector, V[:,l] is a unit vector, W[:,l] is a unit vector.
    # The product U_{u,l} V_{v,l} W_{w,l} is 1 only if u, v, w match the specific indices for that l.
    # Since each l corresponds to a unique (i,j,k) triple, and the tensor T has 1 at exactly those (u,v,w) triples,
    # the sum over l will be 1 for the valid triples and 0 for all others (since no other column l' will have 1s at the same u,v,w).
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
