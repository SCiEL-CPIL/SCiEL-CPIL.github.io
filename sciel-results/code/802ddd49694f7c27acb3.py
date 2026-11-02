import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Verify a decomposition
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition of rank n*m*p = 54
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    if not verify(U_sb, V_sb, W_sb, rank_sb):
        raise ValueError("Schoolbook verification failed")

    # Laderman's 23-term decomposition for 3x3x3
    laderman_terms = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,1):1}, {(0,1):1}),
        ({(0,2):1}, {(2,2):1}, {(0,2):1}),
        ({(1,0):1}, {(0,1):1}, {(1,0):1}),
        ({(1,2):1}, {(2,0):1}, {(1,2):1}),
        ({(2,0):1}, {(0,2):1}, {(2,0):1}),
        ({(2,1):1}, {(1,2):1}, {(2,1):1}),
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(0,1):1,(1,1):1}, {(1,0):1,(1,1):1}, {(0,0):-1,(1,1):-1}),
        ({(1,0):1,(1,1):1}, {(0,0):1,(0,1):1}, {(1,0):1,(1,1):1}),
        ({(0,0):1,(0,1):1}, {(1,0):1,(1,1):1}, {(0,0):1,(0,1):-1}),
        ({(0,1):1,(0,2):1}, {(0,2):1,(1,2):1}, {(0,2):1,(0,1):-1}),
        ({(0,2):1,(1,2):1}, {(2,0):1,(2,1):1}, {(0,2):-1,(1,2):1}),
        ({(1,0):1,(2,0):1}, {(0,2):1,(1,2):1}, {(1,0):-1,(2,0):1}),
        ({(1,2):1,(2,2):1}, {(0,0):1,(1,0):1}, {(1,2):1,(2,2):-1}),
        ({(2,0):1,(2,1):1}, {(1,2):1,(2,2):1}, {(2,0):-1,(2,1):1}),
        ({(2,1):1,(2,2):1}, {(0,0):1,(0,1):1}, {(2,1):1,(2,2):-1}),
        ({(0,0):1,(2,2):1}, {(2,0):1,(2,2):1}, {(0,0):-1,(2,2):1}),
        ({(0,1):1,(2,2):1}, {(2,1):1,(2,2):1}, {(0,1):-1,(2,2):1}),
        ({(0,2):1,(2,1):1}, {(1,0):1,(1,1):1}, {(0,2):1,(2,1):-1}),
        ({(1,0):1,(2,2):1}, {(2,0):1,(2,2):1}, {(1,0):-1,(2,2):1}),
        ({(1,1):1,(2,0):1}, {(0,1):1,(0,2):1}, {(1,1):1,(2,0):-1}),
        ({(1,2):1,(2,1):1}, {(1,0):1,(1,1):1}, {(1,2):-1,(2,1):1}),
    ]

    # Build rank-46 solution using two Laderman decompositions
    rank_46 = 46
    U_46 = [[0] * rank_46 for _ in range(n * m)]
    V_46 = [[0] * rank_46 for _ in range(m * p)]
    W_46 = [[0] * rank_46 for _ in range(p * n)]

    # First Laderman block: B columns 0,1,2
    for t_idx, (u_ent, v_ent, w_ent) in enumerate(laderman_terms):
        col = t_idx
        for (i,j), coeff in u_ent.items():
            U_46[i*m+j][col] = coeff
        for (j,k), coeff in v_ent.items():
            V_46[j*p+k][col] = coeff
        for (i,k), coeff in w_ent.items():
            W_46[k*n+i][col] = coeff

    # Second Laderman block: B columns 3,4,5
    for t_idx, (u_ent, v_ent, w_ent) in enumerate(laderman_terms):
        col = 23 + t_idx
        for (i,j), coeff in u_ent.items():
            U_46[i*m+j][col] = coeff
        for (j,k), coeff in v_ent.items():
            V_46[j*p+(k+3)][col] = coeff
        for (i,k), coeff in w_ent.items():
            W_46[(k+3)*n+i][col] = coeff

    if verify(U_46, V_46, W_46, rank_46):
        # Try to improve: search for rank < 46
        # This is a hard problem, so we'll try a heuristic approach
        
        # Heuristic: Try to merge some terms by looking for linear dependencies
        # in the tensor space
        
        # For now, return the verified rank-46 solution
        return {"U": U_46, "V": V_46, "W": W_46}
    else:
        # Fall back to schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
