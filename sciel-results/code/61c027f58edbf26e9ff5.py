"""Rewritten program for matrix multiplication 2,4,5 rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 5
    # Try to find a low-rank decomposition with half-integer entries
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try structured approaches and random search
    for rank in range(16, 41):
        for trial in range(500):
            np.random.seed(trial * 10000 + rank * 100 + 42)
            U_rand = np.random.randn(n * m, rank)
            V_rand = np.random.randn(m * p, rank)
            
            # Build M: M[(k*n+i), r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p * n, rank))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        M[k * n + i, :] += U_rand[i * m + j, :] * V_rand[j * p + k, :]
            
            # For each output (i,k), need sum_r M[k*n+i, r] * W[k*n+i, r] = 1
            W_sol = np.zeros((p * n, rank))
            valid = True
            for idx in range(p * n):
                row = M[idx, :]
                norm_sq = np.dot(row, row)
                if norm_sq < 1e-8:
                    valid = False
                    break
                W_sol[idx, :] = row / norm_sq
            
            if not valid:
                continue
            
            # Verify decomposition
            C_check = np.ones((n, p))
            ok = True
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        for j in range(m):
                            s += U_rand[i * m + j, r] * V_rand[j * p + k, r] * W_sol[k * n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Try to find a global scaling that makes entries half-integers
            # Check if entries are already close to half-integers
            U_round = np.round(U_rand * 2) / 2
            V_round = np.round(V_rand * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            # Verify rounded version
            ok2 = True
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        for j in range(m):
                            s += U_round[i * m + j, r] * V_round[j * p + k, r] * W_round[k * n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        ok2 = False
                        break
                if not ok2:
                    break
            if ok2:
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
