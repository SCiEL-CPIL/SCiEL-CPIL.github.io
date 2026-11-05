import numpy as np
import time
from itertools import product
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 115  # Safety margin before 120s limit
    
    # Precompute valid and invalid indices for the tensor T
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_indices.append((alpha, beta, gamma))
                
    total_indices = []
    for alpha in range(n * m):
        for beta in range(m * p):
            for gamma in range(p * n):
                total_indices.append((alpha, beta, gamma))
                
    valid_set = set(valid_indices)
    invalid_indices = [idx for idx in total_indices if idx not in valid_set]

    def check_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        for (a, b, g) in valid_indices:
            if not np.isclose(T_approx[a, b, g], 1.0, atol=atol):
                return False
        for (a, b, g) in invalid_indices:
            if not np.isclose(T_approx[a, b, g], 0.0, atol=atol):
                return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap values to nearest half-integer."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def try_known_rank_34():
        rank = 34
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        # ... (Same as before, omitted for brevity in thought, included in final code) ...
        # Constructing the specific terms for the known 34 solution.
        
        # Part 1: A_TL (2x2) * B_top_L (2x2) -> C_top_L (2x2)
        # Strassen 2x2
        U[0, col] += 1; U[4, col] += 1
        V[0, col] += 1; V[5, col] += 1
        W[0, col] += 1; W[4, col] += 1
        col += 1
        
        U[3, col] += 1; U[4, col] += 1
        V[0, col] += 1
        W[1, col] += 1; W[4, col] += -1
        col += 1
        
        U[0, col] += 1
        V[1, col] += 1; V[5, col] += -1
        W[3, col] += 1; W[4, col] += 1
        col += 1
        
        U[4, col] += 1
        V[4, col] += 1; V[0, col] += -1
        W[0, col] += 1; W[1, col] += 1
        col += 1
        
        U[0, col] += 1; U[1, col] += 1
        V[5, col] += 1
        W[3, col] += 1; W[0, col] += -1
        col += 1
        
        U[3, col] += 1; U[0, col] += -1
        V[0, col] += 1; V[1, col] += 1
        W[4, col] += 1
        col += 1
        
        U[1, col] += 1; U[4, col] += -1
        V[4, col] += 1; V[5, col] += 1
        W[0, col] += 1
        col += 1
        
        # Part 2: A_TL (2x2) * B_top_R (2x2) -> C_top_R (2x2)
        U[0, col] += 1; U[4, col] += 1
        V[2, col] += 1; V[6, col] += 1
        W[6, col] += 1; W[9, col] += 1
        col += 1
        
        U[3, col] += 1; U[4, col] += 1
        V[2, col] += 1
        W[7, col] += 1; W[10, col] += -1
        col += 1
        
        U[0, col] += 1
        V[3, col] += 1; V[6, col] += -1
        W[9, col] += 1; W[10, col] += 1
        col += 1
        
        U[4, col] += 1
        V[6, col] += 1; V[2, col] += -1
        W[6, col] += 1; W[7, col] += 1
        col += 1
        
        U[0, col] += 1; U[1, col] += 1
        V[6, col] += 1
        W[9, col] += 1; W[6, col] += -1
        col += 1
        
        U[3, col] += 1; U[0, col] += -1
        V[2, col] += 1; V[3, col] += 1
        W[10, col] += 1
        col += 1
        
        U[1, col] += 1; U[4, col] += -1
        V[6, col] += 1; V[7, col] += 1
        W[6, col] += 1
        col += 1
        
        # Part 3: A_TR (2x1) * B_bot (1x4)
        U[2, col] += 1; V[8, col] += 1; W[0, col] += 1; col += 1
        U[2, col] += 1; V[9, col] += 1; W[3, col] += 1; col += 1
        U[2, col] += 1; V[10, col] += 1; W[6, col] += 1; col += 1
        U[2, col] += 1; V[11, col] += 1; W[9, col] += 1; col += 1
        
        U[5, col] += 1; V[8, col] += 1; W[1, col] += 1; col += 1
        U[5, col] += 1; V[9, col] += 1; W[4, col] += 1; col += 1
        U[5, col] += 1; V[10, col] += 1; W[7, col] += 1; col += 1
        U[5, col] += 1; V[11, col] += 1; W[10, col] += 1; col += 1
        
        # Part 4: A_BL (1x2) * B_top (2x4)
        U[6, col] += 1; V[0, col] += 1; W[2, col] += 1; col += 1
        U[7, col] += 1; V[4, col] += 1; W[2, col] += 1; col += 1
        
        U[6, col] += 1; V[1, col] += 1; W[5, col] += 1; col += 1
        U[7, col] += 1; V[5, col] += 1; W[5, col] += 1; col += 1
        
        U[6, col] += 1; V[2, col] += 1; W[8, col] += 1; col += 1
        U[7, col] += 1; V[6, col] += 1; W[8, col] += 1; col += 1
        
        U[6, col] += 1; V[3, col] += 1; W[11, col] += 1; col += 1
        U[7, col] += 1; V[7, col] += 1; W[11, col] += 1; col += 1
        
        # Part 5: A_BR (1x1) * B_bot (1x4)
        U[8, col] += 1; V[8, col] += 1; W[2, col] += 1; col += 1
        U[8, col] += 1; V[9, col] += 1; W[5, col] += 1; col += 1
        U[8, col] += 1; V[10, col] += 1; W[8, col] += 1; col += 1
        U[8, col] += 1; V[11, col] += 1; W[11, col] += 1; col += 1
        
        if check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    result = try_known_rank_34()
    if result is not None:
        return result

    # Fallback: Schoolbook
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
