import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T
    # T[a, b, c] = 1 if a, b, c correspond to i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomp(U, V, W):
        """
        Checks if U, V, W provide an exact decomposition.
        Attempts to snap entries to the nearest half-integer as per problem constraints.
        """
        # Snap to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Calculate resulting tensor
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        
        # Check for exact match (allowing tiny floating point error)
        if np.allclose(T_calc, T, atol=1e-9):
            return True, U_r, V_r, W_r
        return False, None, None

    def try_find_rank(R, num_restarts=100):
        """
        Uses Gradient Descent with Momentum to find a rank-R decomposition.
        """
        for restart in range(num_restarts):
            # Vary initialization scale to escape local minima
            scales = [0.1, 0.5, 1.0, 2.0, 0.01]
            scale = scales[restart % len(scales)]
            
            U = np.random.randn(nm, R) * scale
            V = np.random.randn(mp, R) * scale
            W = np.random.randn(pn, R) * scale
            
            # Gradient Descent with Momentum
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            # Adaptive learning rate / early stopping
            best_loss = float('inf')
            patience = 0
            max_patience = 200
            
            for step in range(3000):
                # Compute current tensor
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                err = T_calc - T
                loss = np.sum(err**2)
                
                if loss < 1e-8:
                    break
                
                if loss < best_loss:
                    best_loss = loss
                    patience = 0
                else:
                    patience += 1
                    if patience > max_patience:
                        break
                
                # Compute gradients
                # L = sum (T_calc - T)^2
                # dL/dU[a,r] = 2 * sum_b,c err[a,b,c] * V[b,r] * W[c,r]
                gU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
                gV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
                gW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
                
                # Update velocities and parameters
                vU = momentum * vU + lr * gU
                vV = momentum * vV + lr * gV
                vW = momentum * vW + lr * gW
                
                U -= vU
                V -= vV
                W -= vW
                
            # Final check
            ok, U_r, V_r, W_r = verify_decomp(U, V, W)
            if ok:
                return U_r, V_r, W_r
                
        return None

    # Try to find minimal rank
    # Known minimal rank for 3x3 is 23.
    # We try from 23 upwards.
    for R in range(23, 27):
        result = try_find_rank(R, num_restarts=50 if R == 23 else 20)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook (Rank 27)
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
