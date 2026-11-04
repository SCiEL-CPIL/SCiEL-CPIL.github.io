import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find a rank-19 decomposition using randomized search
    # with local refinement
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    for rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Local search: try to reduce the error
            # We'll use a simple gradient-free method
            # Compute initial error
            T_comp = np.einsum('al,bl,cl->abc', U, V, W)
            err = np.sum((T_comp - T)**2)
            
            # Try to improve
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Try random perturbations
                improved = False
                for _ in range(10):
                    # Pick a random entry to perturb
                    mat = np.random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        idx = (np.random.randint(nm), np.random.randint(rank))
                        delta = np.random.choice([-0.5, -0.25, 0.25, 0.5])
                        U_new = U.copy()
                        U_new[idx] += delta
                        T_comp_new = np.einsum('al,bl,cl->abc', U_new, V, W)
                        err_new = np.sum((T_comp_new - T)**2)
                        if err_new < err:
                            U = U_new
                            err = err_new
                            improved = True
                    elif mat == 'V':
                        idx = (np.random.randint(mp), np.random.randint(rank))
                        delta = np.random.choice([-0.5, -0.25, 0.25, 0.5])
                        V_new = V.copy()
                        V_new[idx] += delta
                        T_comp_new = np.einsum('al,bl,cl->abc', U, V_new, W)
                        err_new = np.sum((T_comp_new - T)**2)
                        if err_new < err:
                            V = V_new
                            err = err_new
                            improved = True
                    else:
                        idx = (np.random.randint(pn), np.random.randint(rank))
                        delta = np.random.choice([-0.5, -0.25, 0.25, 0.5])
                        W_new = W.copy()
                        W_new[idx] += delta
                        T_comp_new = np.einsum('al,bl,cl->abc', U, V, W_new)
                        err_new = np.sum((T_comp_new - T)**2)
                        if err_new < err:
                            W = W_new
                            err = err_new
                            improved = True
                
                if not improved:
                    break
            
            # Check if we found an exact solution
            if err < 1e-10:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_comp_r = np.einsum('al,bl,cl->abc', U_r, V_r, W_r)
                if np.max(np.abs(T_comp_r - T)) < 1e-10:
                    best_result = (rank, U_r, V_r, W_r)
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        rank, U, V, W = best_result
        U_list = [[float(U[i][j]) for j in range(rank)] for i in range(nm)]
        V_list = [[float(V[i][j]) for j in range(rank)] for i in range(mp)]
        W_list = [[float(W[i][j]) for j in range(rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook decomposition
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
