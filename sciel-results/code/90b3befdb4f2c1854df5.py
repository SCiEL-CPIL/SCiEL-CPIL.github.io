import numpy as np

def construct():
    n, m, p = 2, 2, 7
    # Try Rank 24 first
    rank = 24
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, r):
        U[u_idx, r] += 1
        V[v_idx, r] += 1
        W[w_idx, r] += 1

    # Block 1: Cols 0,1 (Strassen)
    # ... (standard Strassen logic) ...
    
    # Block 2: Cols 2,3 (Strassen)
    # ...
    
    # Block 3: Cols 4,5,6 (Rank 10)
    # ...
    
    # Verify
    # If valid, return Rank 24
    # Else, return Rank 25 (current logic)
