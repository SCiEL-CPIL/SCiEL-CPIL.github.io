"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Rank-8 decomposition using Strassen for (2,2,2) part + rank-1 for third column
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    rank = 8
    
    # U: 4 x 8
    U = [
        [1,  0,  1,  0,  1, -1,  0,  1],  # i=0,j=0
        [0,  0,  0,  0,  1,  0,  1,  1],  # i=0,j=1
        [0,  1,  0,  0,  0,  1,  0,  1],  # i=1,j=0
        [1,  1,  0,  1,  0,  0, -1,  1],  # i=1,j=1
    ]
    
    # V: 6 x 8
    V = [
        [1,  1,  0, -1,  0,  1,  0,  0],  # j=0,k=0
        [0,  0,  1,  0,  0,  1,  0,  0],  # j=0,k=1
        [0,  0,  0,  0,  0,  0,  0,  1],  # j=0,k=2
        [1,  0,  0,  1,  0,  0,  1,  0],  # j=1,k=0
        [0,  0, -1, 0,  1,  0,  0,  1],  # j=1,k=1
        [0,  0,  0,  0,  0,  0,  1,  0],  # j=1,k=2
    ]
    
    # W: 6 x 8 (indexed by k*n+i)
    W = [
        [1,  0,  0,  1,  0,  0,  1,  0],  # k=0,i=0
        [0,  1,  0,  1,  0,  0,  1,  0],  # k=0,i=1
        [0,  0,  1,  0,  1,  0,  0,  0],  # k=1,i=0
        [1,  0,  0,  0,  1,  1,  0,  0],  # k=1,i=1
        [0,  0,  0,  0,  0,  0,  0,  1],  # k=2,i=0
        [0,  0,  0,  0,  0,  0,  0,  1],  # k=2,i=1
    ]
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    
    # Check that T matches the expected tensor
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-10:
                    max_err = max(max_err, abs(T[a, b, c] - 1.0))
    
    # Check that all other entries are 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Check if (a,b,c) corresponds to a valid (i,j,k)
                valid = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if a == i*m+j and b == j*p+k and c == k*n+i:
                                valid = True
                if not valid and abs(T[a, b, c]) > 1e-10:
                    max_err = max(max_err, abs(T[a, b, c]))
    
    if max_err > 1e-8:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][column] = 1
                    V_school[j * p + k][column] = 1
                    W_school[k * n + i][column] = 1
                    column += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
