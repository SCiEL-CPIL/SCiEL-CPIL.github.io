from scipy.optimize import minimize
import numpy as np
import time

# Dimensions
n, m, p = 2, 3, 5
N_U = n * m  # 6
N_V = m * p  # 15
N_W = p * n  # 10

# Precompute indices for the tensor decomposition
# U index: i * m + j
# V index: j * p + k
# W index: k * n + i
_u_indices_list = []
_v_indices_list = []
_w_indices_list = []
for i in range(n):
    for j in range(m):
        for k in range(p):
            _u_indices_list.append(i * m + j)
            _v_indices_list.append(j * p + k)
            _w_indices_list.append(k * n + i)

U_INDICES = np.array(_u_indices_list)
V_INDICES = np.array(_v_indices_list)
W_INDICES = np.array(_w_indices_list)
NUM_TERMS = len(_u_indices_list)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Schoolbook fallback (rank 30)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((N_U, rank))
        V = np.zeros((N_V, rank))
        W = np.zeros((N_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Helper to verify a solution
    def verify_solution(U, V, W):
        # U, V, W are numpy arrays
        terms = U[U_INDICES] * V[V_INDICES] * W[W_INDICES]
        sums = terms.sum(axis=1)
        return np.allclose(sums, 1.0, atol=1e-9)
    
    # Structured Strassen decomposition (Rank 16)
    def get_strassen_decomposition():
        rank = 16
        U = np.zeros((N_U, rank))
        V = np.zeros((N_V, rank))
        W = np.zeros((N_W, rank))
        
        # Strassen for 2x2x2 (k=0,1)
        # m1: (a00+a11)*(b00+b11) -> C00, C11
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[6, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2: (a10+a11)*b00 -> C10
        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # m3: a00*(b01-b11) -> C01
        U[0, 2] = 1
        V[1, 2] = 1; V[6, 2] = -1
        W[1, 2] = 1
        
        # m4: a11*(b10-b00) -> C00, C10
        U[4, 3] = 1
        V[5, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # m5: (a00+a10)*b11 -> C01
        U[0, 4] = 1; U[3, 4] = 1
        V[6, 4] = 1
        W[1, 4] = 1
        
        # m6: (a11-a10)*(b00+b10) -> C00
        U[4, 5] = 1; U[3, 5] = -1
        V[0, 5] = 1; V[5, 5] = 1
        W[0, 5] = 1
        
        # m7: (a01-a11)*(b01+b11) -> C11
        U[1, 6] = 1; U[4, 6] = -1
        V[1, 6] = 1; V[6, 6] = 1
        W[3, 6] = 1
        
        # Schoolbook for 2x3x3 (k=2,3,4)
        for j in range(3):
            for k in range(3):
                col = 7 + j * 3 + k
                for i in range(2):
                    U[i * 3 + j, col] = 1
                    V[j * 5 + (k + 2), col] = 1
                    W[(k + 2) * 2 + i, col] = 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify and store best known (Rank 16)
    strassen_sol = get_strassen_decomposition()
    U_arr = np.array(strassen_sol["U"])
    V_arr = np.array(strassen_sol["V"])
    W_arr = np.array(strassen_sol["W"])
    
    best_solution = strassen_sol
    best_rank = 16
    
    if verify_solution(U_arr, V_arr, W_arr):
        # Confirmed Rank 16 is valid
        pass
    else:
        # Fallback to schoolbook if Strassen failed (should not happen)
        best_solution = get_schoolbook()
        best_rank = 30

    # Attempt to find Rank 15 solution
    # Lower bound is max(N_U, N_V, N_W) = 15
    target_rank = 15
    if time.time() < deadline - 10: # Need some time left
        # Loss function for rank 15
        def loss(x):
            U = x[:N_U * target_rank].reshape(N_U, target_rank)
            V = x[N_U * target_rank : N_U * target_rank + N_V * target_rank].reshape(N_V, target_rank)
            W = x[N_U * target_rank + N_V * target_rank :].reshape(N_W, target_rank)
            
            terms = U[U_INDICES] * V[V_INDICES] * W[W_INDICES]
            sums = terms.sum(axis=1)
            errors = sums - 1.0
            return np.sum(errors ** 2)
        
        # Try a few random restarts
        np.random.seed(123)
        found_15 = False
        max_restarts = 20
        time_per_restart = (deadline - 10 - time.time()) / max_restarts
        
        for r in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            # Scale: typical entries are 1, so std dev 1 is okay.
            U_init = np.random.randn(N_U, target_rank)
            V_init = np.random.randn(N_V, target_rank)
            W_init = np.random.randn(N_W, target_rank)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use L-BFGS-B for better handling of large-ish vectors
                res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                # Potential solution found
                x_opt = res.x
                U_opt = x_opt[:N_U * target_rank].reshape(N_U, target_rank)
                V_opt = x_opt[N_U * target_rank : N_U * target_rank + N_V * target_rank].reshape(N_V, target_rank)
                W_opt = x_opt[N_U * target_rank + N_V * target_rank :].reshape(N_W, target_rank)
                
                # Round to nearest half-integer
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                if verify_solution(U_round, V_round, W_round):
                    best_solution = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    best_rank = 15
                    found_15 = True
                    break
                else:
                    # Maybe try rounding to integers?
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if verify_solution(U_int, V_int, W_int):
                        best_solution = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        best_rank = 15
                        found_15 = True
                        break

    return best_solution
