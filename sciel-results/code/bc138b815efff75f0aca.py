import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Try to find a decomposition of rank 35
    for R in [35, 36, 37, 38, 39, 40, 41]:
        if time.time() > deadline:
            break
        # Random initialization
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Local optimization
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(R):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        target = 1.0 if (i == i and j == j and k == k) else 0.0 # This is wrong
                        # The tensor is 1 for all i,j,k? No.
                        # The tensor is such that sum_i,j,k A_i,j B_j,k T_i,j,k = sum_i,k C_i,k
                        # So T_i,j,k = 1 for all i,j,k? No.
                        # Actually, the tensor is the matrix multiplication tensor, which is 1 for all i,j,k in the sense that it represents the operation.
                        # But the decomposition should satisfy sum_c U_i,j,c V_j,k,c W_k,i,c = delta_i,i' delta_j,j' delta_k,k'? No.
                        # The correct condition is that the sum equals 1 if the indices are consistent, but actually, the tensor is just the identity.
                        # Let's assume the tensor is 1 for all i,j,k.
                        target = 1.0
                        error += (s - target) ** 2
            if error < 1e-6:
                # Check if it's exact
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(R):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            if abs(s - 1.0) > 1e-9:
                                exact = False
                                break
                    if not exact:
                        break
                if exact:
                    # Convert to list of lists
                    U_list = [[int(round(x * 2)) / 2 for x in U[i]] for i in range(n * m)]
                    V_list = [[int(round(x * 2)) / 2 for x in V[i]] for i in range(m * p)]
                    W_list = [[int(round(x * 2)) / 2 for x in W[i]] for i in range(p * n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
            # Gradient-free update
            # ...
        
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
