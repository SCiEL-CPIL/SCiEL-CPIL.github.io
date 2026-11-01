import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    nr, mr, pr = n*m, m*p, p*n  # 4, 16, 16
    
    # Build the target tensor
    T = np.zeros((nr, mr, pr), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-8):
        T_calc = np.einsum('at,bt,ct->abc', U, V, W)
        err = np.max(np.abs(T_calc - T))
        return err < tol, err
    
    def get_strassen_28():
        """Standard Strassen-based rank-28 decomposition."""
        rank = 28
        U = np.zeros((nr, rank), dtype=float)
        V = np.zeros((mr, rank), dtype=float)
        W = np.zeros((pr, rank), dtype=float)
        
        # Strassen 7-term decomposition for 2x2x2
        # Each term: (A_coeffs, B_coeffs, C_coeffs)
        # A: {(i,j): coeff}, B: {(j,k): coeff}, C: {(i,k): coeff}
        strassen = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for q in range(p // 2):
            k1 = 2 * q
            for r_idx, (A_c, B_c, C_c) in enumerate(strassen):
                t = q * 7 + r_idx
                for (i, j), coeff in A_c.items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in B_c.items():
                    k_global = k1 + k_sub
                    V[j * p + k_global, t] = coeff
                for (i, k_sub), coeff in C_c.items():
                    k_global = k1 + k_sub
                    W[k_global * n + i, t] = coeff
        
        return U, V, W
    
    # Start with rank-28 Strassen solution
    U28, V28, W28 = get_strassen_28()
    ok, err = verify(U28, V28, W28)
    
    # Try to find a lower rank decomposition
    best_rank = 28
    best_U, best_V, best_W = U28, V28, W28
    
    deadline = time.time() + 140.0  # Leave 10s margin
    
    # Strategy: Try rank 27 with multiple random restarts and gradient descent
    for target_rank in [27, 26, 25]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + target_rank * 100 + restart)
            
            # Initialize by taking a subset of the rank-28 solution and perturbing
            U_init = U28[:, :target_rank].copy() + np.random.randn(nr, target_rank) * 0.1
            V_init = V28[:, :target_rank].copy() + np.random.randn(mr, target_rank) * 0.1
            W_init = W28[:, :target_rank].copy() + np.random.randn(pr, target_rank) * 0.1
            
            U = U_init
            V = V_init
            W = W_init
            
            best_loss = np.inf
            best_U_r, best_V_r, best_W_r = U.copy(), V.copy(), W.copy()
            
            max_iters = 200
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                T_calc = np.einsum('at,bt,ct->abc', U, V, W)
                err_tensor = T_calc - T
                loss = np.sum(err_tensor**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U_r, best_V_r, best_W_r = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-12:
                    break
                
                # Gradients
                gU = np.einsum('abc,bt,ct->at', err_tensor, V, W)
                gV = np.einsum('abc,at,ct->bt', err_tensor, U, W)
                gW = np.einsum('abc,at,bt->ct', err_tensor, U, V)
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
                if grad_norm < 1e-15:
                    break
                
                lr = 0.005
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
            
            ok_r, err_r = verify(best_U_r, best_V_r, best_W_r)
            if ok_r:
                best_rank = target_rank
                best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
                found = True
                break
        
        if found:
            break
    
    # If we found a solution, try to round to half-integers
    if best_rank < 28:
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        ok_r, _ = verify(U_round, V_round, W_round)
        if ok_r:
            best_U, best_V, best_W = U_round, V_round, W_round
        
        if not ok_r:
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            ok_i, _ = verify(U_int, V_int, W_int)
            if ok_i:
                best_U, best_V, best_W = U_int, V_int, W_int
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
