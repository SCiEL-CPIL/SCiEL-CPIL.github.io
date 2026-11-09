import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def _solve_subproblem(n, m, p, target_rank, max_time=10):
    """
    Attempts to find a rank decomposition for M_{n,m,p}.
    Returns (U, V, W) or None.
    """
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                triples.append((u, v, w))
                
    num_triples = len(triples)
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_size = size_u + size_v + size_w
    
    def unpack(x):
        U = x[:size_u].reshape(n*m, target_rank)
        V = x[size_u:size_u+size_v].reshape(m*p, target_rank)
        W = x[size_u+size_v:].reshape(p*n, target_rank)
        return U, V, W

    def residuals(x):
        U, V, W = unpack(x)
        res = np.zeros(num_triples)
        for idx, (u, v, w) in enumerate(triples):
            res[idx] = np.sum(U[u, :] * V[v, :] * W[w, :]) - 1.0
        return res

    def jacobian(x):
        U, V, W = unpack(x)
        J = np.zeros((num_triples, total_size))
        for idx, (u, v, w) in enumerate(triples):
            J[idx, u*target_rank : (u+1)*target_rank] = V[v, :] * W[w, :]
            J[idx, size_u + v*target_rank : size_u + (v+1)*target_rank] = U[u, :] * W[w, :]
            J[idx, size_u + size_v + w*target_rank : size_u + size_v + (w+1)*target_rank] = U[u, :] * V[v, :]
        return J

    start_time = time.time()
    # Try a few random initializations
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        x0 = np.random.choice([-1, -0.5, 0, 0.5, 1], size=total_size)
        try:
            sol = least_squares(residuals, x0, jac=jacobian, method='lm')
            if np.linalg.norm(sol.fun) < 1e-6:
                U, V, W = unpack(sol.x)
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                valid = True
                for u, v, w in triples:
                    val = np.sum(U_round[u, :] * V_round[v, :] * W_round[w, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        valid = False
                        break
                if valid:
                    return U_round, V_round, W_round
        except Exception:
            continue
    return None

def construct():
    n, m, p = DIMENSIONS
    # p=4. Split into p=3 and p=1.
    # M_{2,3,4} = M_{2,3,3} (for k=0,1,2) + M_{2,3,1} (for k=3)
    
    # 1. Solve M_{2,3,3}
    # Try ranks 10, 11, 12
    sol_233 = None
    for r in [10, 11, 12]:
        sol_233 = _solve_subproblem(2, 3, 3, r, max_time=20)
        if sol_233 is not None:
            break
            
    if sol_233 is not None:
        U_233, V_233, W_233 = sol_233
        r_233 = U_233.shape[1]
        
        # 2. Construct M_{2,3,1} solution for k=3
        # This is rank 6.
        r_231 = 6
        U_231 = np.zeros((2*3, r_231))
        V_231 = np.zeros((3*1, r_231)) # Note: in subproblem p=1, so V is 3x1
        W_231 = np.zeros((1*2, r_231))
        
        # Map indices for the full problem
        # u = i*3 + j
        # v_full = j*4 + 3
        # w_full = 3*2 + i = 6 + i
        
        for i in range(2):
            for j in range(3):
                col = i * 3 + j
                U_231[i*3 + j, col] = 1.0
                # V_231 index is j (since p=1 in subproblem, v=j*1+0=j)
                # But in full problem, v is j*4+3.
                # We are building the full matrices, so we need to place them correctly.
                # Let's build the full matrices directly.
        
        # Actually, it's easier to build the full matrices U, V, W of rank r_233 + r_231.
        R = r_233 + r_231
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Copy M_{2,3,3} part
        # U_233 is 6 x r_233. U is 6 x R.
        U[:, :r_233] = U_233
        # V_233 is 9 x r_233 (m=3, p=3). V is 12 x R.
        # V_233 rows correspond to j*3+k for k in 0,1,2.
        # In full V, rows are j*4+k.
        # So V_233[j*3+k] goes to V[j*4+k].
        for j in range(3):
            for k in range(3):
                V[j*4 + k, :r_233] = V_233[j*3 + k, :]
        # W_233 is 6 x r_233 (p=3, n=2). W is 8 x R.
        # W_233 rows correspond to k*2+i for k in 0,1,2.
        # In full W, rows are k*2+i.
        # So W_233[k*2+i] goes to W[k*2+i].
        W[:6, :r_233] = W_233
        
        # Add M_{2,3,1} part (rank 6)
        # This part only affects k=3.
        # u = i*3+j
        # v_full = j*4+3
        # w_full = 3*2+i = 6+i
        for i in range(2):
            for j in range(3):
                col = r_233 + i*3 + j
                U[i*3 + j, col] = 1.0
                V[j*4 + 3, col] = 1.0
                W[6 + i, col] = 1.0
                
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    val = np.sum(U[u, :] * V[v, :] * W[w, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
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
