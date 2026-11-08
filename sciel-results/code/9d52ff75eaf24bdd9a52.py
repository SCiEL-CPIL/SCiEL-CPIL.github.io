import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    num_u = n * m  # 6
    num_v = m * p  # 27
    num_w = p * n  # 18
    
    # Build the target tensor T
    # T[u, v, w] = 1 if indices match, else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((num_u, num_v, num_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    # Try to find a decomposition with rank R
    # We use a heuristic: fix V and W, solve for U
    # If U is integer, we have a candidate.
    
    def try_rank(R, num_restarts=100):
        for _ in range(num_restarts):
            # Initialize V and W with random sparse integers
            # Using -1, 0, 1 with low probability of non-zero
            prob = 0.3
            V = np.random.choice([-1, 0, 1], size=(num_v, R), p=[0.15, 0.7, 0.15])
            W = np.random.choice([-1, 0, 1], size=(num_w, R), p=[0.15, 0.7, 0.15])
            
            U = np.zeros((num_u, R), dtype=int)
            success = True
            
            for u in range(num_u):
                # Construct linear system A * U_u = b
                # A has shape (num_v * num_w, R)
                # A[v * num_w + w, r] = V[v, r] * W[w, r]
                A = np.zeros((num_v * num_w, R), dtype=int)
                b = np.zeros(num_v * num_w, dtype=int)
                
                for v in range(num_v):
                    for w in range(num_w):
                        idx = v * num_w + w
                        for r in range(R):
                            A[idx, r] = V[v, r] * W[w, r]
                        b[idx] = T[u, v, w]
                
                # Solve A * U_u = b
                # Use least squares to check consistency
                try:
                    # If R is small, we can use exact integer logic or np.linalg.lstsq
                    # Since we want integer solutions, we first check if a real solution exists
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Check if residuals are close to 0
                    if residuals and residuals[0] > 1e-6:
                        success = False
                        break
                        
                    # Check if solution is close to integers
                    sol_int = np.round(sol)
                    if np.allclose(A @ sol_int, b, atol=1e-6):
                        U[u] = sol_int
                    else:
                        success = False
                        break
                except Exception:
                    success = False
                    break
            
            if success:
                # Verify the full decomposition
                # C[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
                C = np.zeros((num_u, num_v, num_w), dtype=int)
                for r in range(R):
                    C += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
                
                if np.array_equal(C, T):
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return U_list, V_list, W_list
                    
    # Try ranks from 40 upwards
    for R in range(40, 51):
        result = try_rank(R, num_restarts=50)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback to the original solution if heuristic fails
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def aidx(i, j): return i * m + j
    def bidx(j, k): return j * p + k
    def cidx(k, i): return k * n + i

    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = bidx(0, k0)
        v_f = bidx(0, k1)
        v_g = bidx(1, k0)
        v_h = bidx(1, k1)
        u_a = aidx(0, 0)
        u_b = aidx(0, 1)
        u_c = aidx(1, 0)
        u_d = aidx(1, 1)

        U[u_a][col] += 1.0; U[u_d][col] += 1.0
        V[v_e][col] += 1.0; V[v_h][col] += 1.0
        U[u_c][col+1] += 1.0; U[u_d][col+1] += 1.0
        V[v_e][col+1] += 1.0
        U[u_a][col+2] += 1.0
        V[v_f][col+2] += 1.0; V[v_h][col+2] -= 1.0
        U[u_d][col+3] += 1.0
        V[v_g][col+3] += 1.0; V[v_e][col+3] -= 1.0
        U[u_a][col+4] += 1.0; U[u_b][col+4] += 1.0
        V[v_h][col+4] += 1.0
        U[u_c][col+5] += 1.0; U[u_a][col+5] -= 1.0
        V[v_e][col+5] += 1.0; V[v_f][col+5] += 1.0
        U[u_b][col+6] += 1.0; U[u_d][col+6] -= 1.0
        V[v_g][col+6] += 1.0; V[v_h][col+6] += 1.0

        w_00 = cidx(k0, 0)
        w_01 = cidx(k1, 0)
        w_10 = cidx(k0, 1)
        w_11 = cidx(k1, 1)

        W[w_00][col]   += 1.0
        W[w_00][col+3] += 1.0
        W[w_00][col+4] -= 1.0
        W[w_00][col+6] += 1.0
        W[w_01][col+2] += 1.0
        W[w_01][col+4] += 1.0
        W[w_10][col+1] += 1.0
        W[w_10][col+3] += 1.0
        W[w_11][col]   += 1.0
        W[w_11][col+1] -= 1.0
        W[w_11][col+2] += 1.0
        W[w_11][col+5] += 1.0

        col += 7

    k8 = 8
    v_e = bidx(0, k8)
    v_g = bidx(1, k8)
    u_a = aidx(0, 0); u_b = aidx(0, 1); u_c = aidx(1, 0); u_d = aidx(1, 1)

    U[u_a][col]   += 1.0; V[v_e][col]   += 1.0
    U[u_b][col+1] += 1.0; V[v_g][col+1] += 1.0
    U[u_c][col+2] += 1.0; V[v_e][col+2] += 1.0
    U[u_d][col+3] += 1.0; V[v_g][col+3] += 1.0

    w_08 = cidx(k8, 0)
    w_18 = cidx(k8, 1)
    W[w_08][col]   += 1.0; W[w_08][col+1] += 1.0
    W[w_18][col+2] += 1.0; W[w_18][col+3] += 1.0
    col += 4

    u_ar0 = aidx(0, 2)
    u_ar1 = aidx(1, 2)
    for k in range(9):
        v_b = bidx(2, k)
        U[u_ar0][col]   += 1.0; V[v_b][col]   += 1.0
        W[cidx(k, 0)][col] += 1.0
        U[u_ar1][col+1] += 1.0; V[v_b][col+1] += 1.0
        W[cidx(k, 1)][col+1] += 1.0
        col += 2

    U = [[int(round(v)) for v in row] for row in U]
    V = [[int(round(v)) for v in row] for row in V]
    W = [[int(round(v)) for v in row] for row in W]

    return {"U": U, "V": V, "W": W}
