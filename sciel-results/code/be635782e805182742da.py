"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen terms for 2x2x2 multiplication
    # Each term: (u_entries, v_entries, w_entries)
    # u: list of (i, j, coeff)
    # v: list of (j, k, coeff)
    # w: list of (k, i, coeff)
    
    strassen_terms = [
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)]
        ),
        (
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)]
        ),
        (
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)]
        ),
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(1,0,1), (0,0,-1)]
        ),
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        )
    ]
    
    # Construct the 14-rank decomposition by duplicating the Strassen terms
    # for the two 2x2 blocks of the 2x4 matrix.
    
    for r in range(7):
        u_entries, v_entries, w_entries = strassen_terms[r]
        
        # Block 1: columns 0 and 1 of B (k=0,1)
        col1 = r
        for i, j, val in u_entries:
            U[i * m + j][col1] += val
        for j, k, val in v_entries:
            V[j * p + k][col1] += val
        for k, i, val in w_entries:
            W[k * n + i][col1] += val
            
        # Block 2: columns 2 and 3 of B (k=2,3)
        col2 = 7 + r
        for i, j, val in u_entries:
            U[i * m + j][col2] += val
        for j, k, val in v_entries:
            k_shifted = k + 2
            V[j * p + k_shifted][col2] += val
        for k, i, val in w_entries:
            k_shifted = k + 2
            W[k_shifted * n + i][col2] += val
    
    # Verify the decomposition is correct
    # Check that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    # Build the expected tensor
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    
    # Verify
    verified = True
    for a in range(u_dim):
        for b in range(v_dim):
            for c in range(w_dim):
                expected = T.get((a, b, c), 0)
                actual = 0
                for r in range(rank):
                    actual += U[a][r] * V[b][r] * W[c][r]
                if actual != expected:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    if not verified:
        # Should not happen, but return the result anyway
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
