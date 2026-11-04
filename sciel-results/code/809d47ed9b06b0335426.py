import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c]."""
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _reconstruction(U, V, W):
    """Compute the tensor reconstructed from factor matrices."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _gradients(U, V, W, T):
    """Compute gradients of the squared error with respect to U, V, W."""
    R = _reconstruction(U, V, W)
    D = T - R
    # grad_U[a, r] = -2 * sum_{b,c} D[a,b,c] V[b,r] W[c,r]
    G_U = -2.0 * np.einsum('abc,br,cr->ar', D, V, W)
    G_V = -2.0 * np.einsum('abc,ar,cr->br', D, U, W)
    G_W = -2.0 * np.einsum('abc,ar,br->cr', D, U, V)
    return G_U, G_V, G_W, np.sum(D*D)

def _snap_to_half_integers(U, V, W, T, tol=1e-9):
    """Try to snap to half-integers and verify exactness."""
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    T_comp = _reconstruction(U_r, V_r, W_r)
    if np.max(np.abs(T_comp - T)) < tol:
        return U_r, V_r, W_r
    return None

def _try_rank(rank, T, nm, mp, pn, deadline, n_restarts=50):
    """Attempt to find a rank-`rank` decomposition via gradient descent."""
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        # Scale matters. If we start too large, gradients are huge.
        # If too small, we move slowly.
        scale = 0.5
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        
        lr = 0.01
        momentum_U = np.zeros_like(U)
        momentum_V = np.zeros_like(V)
        momentum_W = np.zeros_like(W)
        beta = 0.9
        
        err = np.inf
        for step in range(2000):
            if time.time() > deadline:
                break
            
            G_U, G_V, G_W, err = _gradients(U, V, W, T)
            
            if err < 1e-10:
                break
            
            # Update momentum
            momentum_U = beta * momentum_U - lr * G_U
            momentum_V = beta * momentum_V - lr * G_V
            momentum_W = beta * momentum_W - lr * G_W
            
            U += momentum_U
            V += momentum_V
            W += momentum_W
            
            # Adaptive learning rate: if error increases, reduce lr
            if err > prev_err:
                lr *= 0.9
            else:
                lr = min(lr * 1.01, 0.1)
            prev_err = err
        
        # Check for exact solution
        if err < 1e-8:
            snapped = _snap_to_half_integers(U, V, W, T)
            if snapped is not None:
                U_r, V_r, W_r = snapped
                return (U_r, V_r, W_r)
        
        # Try a few more steps with smaller lr if close
        if err < 1e-4:
             for step in range(500):
                 if time.time() > deadline:
                     break
                 G_U, G_V, G_W, err = _gradients(U, V, W, T)
                 if err < 1e-10:
                     break
                 lr = 0.001
                 U -= lr * G_U
                 V -= lr * G_V
                 W -= lr * G_W
             
             if err < 1e-8:
                 snapped = _snap_to_half_integers(U, V, W, T)
                 if snapped is not None:
                     U_r, V_r, W_r = snapped
                     return (U_r, V_r, W_r)

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    T = _build_tensor(n, m, p)
    
    # Try ranks from 20 upward. 19 is very hard to find randomly.
    # 20 is a strong candidate.
    for rank in [20, 21, 22, 23]:
        if time.time() > deadline:
            break
        
        result = _try_rank(rank, T, nm, mp, pn, deadline, n_restarts=30)
        if result is not None:
            U_r, V_r, W_r = result
            U_list = [[float(U_r[i][j]) for j in range(rank)] for i in range(nm)]
            V_list = [[float(V_r[i][j]) for j in range(rank)] for i in range(mp)]
            W_list = [[float(W_r[i][j]) for j in range(rank)] for i in range(pn)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook decomposition as fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
