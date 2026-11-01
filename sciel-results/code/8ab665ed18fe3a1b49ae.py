import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook U, V, W
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    def to_tensor(U, V, W):
        # U: n*m x rank
        # V: m*p x rank
        # W: p*n x rank
        # Tensor T: (n*m, m*p, p*n)
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        rank = len(U[0])
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = np.array([U[i][r] for i in range(n*m)])
            v = np.array([V[j][r] for j in range(m*p)])
            w = np.array([W[k][r] for k in range(p*n)])
            T += np.einsum('a,b,c->abc', u, v, w)
        return T

    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1

    # Check if a decomposition is valid
    def check_decomposition(U, V, W):
        T = to_tensor(U, V, W)
        return np.allclose(T, T_target, atol=1e-9)

    # Search for rank 17
    target_rank = 17
    if target_rank < rank_schoolbook:
        start_time = time.time()
        time_limit = 10.0 # seconds
        
        # Initialize with schoolbook, drop one term, perturb others
        # We try dropping each term and seeing if we can adjust
        for drop_idx in range(rank_schoolbook):
            if time.time() - start_time > time_limit:
                break
                
            # Create initial guess
            U_init = []
            V_init = []
            W_init = []
            for r in range(rank_schoolbook):
                if r == drop_idx:
                    continue
                U_init.append([U_sb[i][r] for i in range(n*m)])
                V_init.append([V_sb[j][r] for j in range(m*p)])
                W_init.append([W_sb[k][r] for k in range(p*n)])
            
            # Convert to numpy for easier manipulation
            U_arr = np.array(U_init, dtype=float)
            V_arr = np.array(V_init, dtype=float)
            W_arr = np.array(W_init, dtype=float)
            
            # Add small random noise
            U_arr += np.random.normal(0, 0.1, U_arr.shape)
            V_arr += np.random.normal(0, 0.1, V_arr.shape)
            W_arr += np.random.normal(0, 0.1, W_arr.shape)
            
            # Simple local search (gradient descent is complex, use random steps)
            best_err = np.inf
            best_U, best_V, best_W = U_arr, V_arr, W_arr
            
            for step in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Calculate error
                T_curr = to_tensor_from_numpy(U_arr, V_arr, W_arr)
                err = np.linalg.norm(T_curr - T_target)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
                    
                    if err < 1e-9:
                        # Found! Convert to list of lists of ints/floats
                        # Check if they are close to half-integers
                        U_final = [[0]*target_rank for _ in range(n*m)]
                        V_final = [[0]*target_rank for _ in range(m*p)]
                        W_final = [[0]*target_rank for _ in range(p*n)]
                        
                        for i in range(n*m):
                            for r in range(target_rank):
                                val = best_U[i][r]
                                # Round to nearest 0.5
                                rounded = round(val * 2) / 2
                                U_final[i][r] = rounded
                                
                        for j in range(m*p):
                            for r in range(target_rank):
                                val = best_V[j][r]
                                rounded = round(val * 2) / 2
                                V_final[j][r] = rounded
                                
                        for k in range(p*n):
                            for r in range(target_rank):
                                val = best_W[k][r]
                                rounded = round(val * 2) / 2
                                W_final[k][r] = rounded
                                
                        # Verify the rounded solution
                        if check_decomposition(U_final, V_final, W_final):
                            return {"U": U_final, "V": V_final, "W": W_final}

                # Perturb
                U_arr += np.random.normal(0, 0.1, U_arr.shape)
                V_arr += np.random.normal(0, 0.1, V_arr.shape)
                W_arr += np.random.normal(0, 0.1, W_arr.shape)

    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}

def to_tensor_from_numpy(U, V, W):
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        T += np.einsum('a,b,c->abc', u, v, w)
    return T
