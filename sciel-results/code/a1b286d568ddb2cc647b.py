import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    # First, try to find a rank-16 decomposition
    result_16 = _try_rank16(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Fallback: rank-17 decomposition
    return _rank17_decomposition(n, m, p)

def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # Block 1: Strassen 2x2x2 (j in {0,1}, k in {0,1})
    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    # Block 2: Direct 2x2x1 (j in {0,1}, k=2)
    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    # Block 3: Direct 2x1x3 (j=2, k in {0,1,2})
    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }

def _check_exact(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition is exact (half-integer entries)."""
    # Round to half-integers
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    # Check if the rounded values actually produce the tensor
    # Convert to integer grid (multiply by 2) for exact integer arithmetic check
    Ui = np.rint(U_s * 2).astype(int)
    Vi = np.rint(V_s * 2).astype(int)
    Wi = np.rint(W_s * 2).astype(int)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    # The product in the integer domain will be 4 * (original product)
    # Original product should be 1. So integer product should be 4.
    # Wait, U, V, W are half-integers.
    # Let U = Ui/2, V = Vi/2, W = Wi/2.
    # Sum (Ui/2) (Vi/2) (Wi/2) = 1/8 Sum Ui Vi Wi.
    # So Sum Ui Vi Wi should be 8.
    
    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None

def _try_rank16(n, m, p, deadline):
    rank = 16
    
    # We will try multiple random initializations
    # To increase chances, we use a structured initialization based on the rank-17 solution
    # by randomly perturbing it and then optimizing.
    
    # Get the rank 17 solution to use as a base for perturbation
    base_17 = _rank17_decomposition(n, m, p)
    U_base = np.array(base_17["U"])
    V_base = np.array(base_17["V"])
    W_base = np.array(base_17["W"])
    
    # We need to reduce rank from 17 to 16.
    # We can try to drop one column and re-optimize, or just random start.
    # Random start is more robust for finding global minima in this small space.
    
    best_err = float('inf')
    best_state = None
    
    for seed in range(200): # Increased seeds
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Optimization loop
        lr = 0.05
        # Use Adam-like momentum for better convergence
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        beta1 = 0.9
        
        for step in range(1000):
            if time.time() > deadline:
                break
                
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            err = np.sum(res**2)
            
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
                
            if err < 1e-8:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
            
            # Gradients
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            # Simple gradient step with adaptive lr
            # Normalize gradients to prevent explosion
            norm_U = np.linalg.norm(dU)
            norm_V = np.linalg.norm(dV)
            norm_W = np.linalg.norm(dW)
            
            if norm_U > 1e-12: dU /= norm_U
            if norm_V > 1e-12: dV /= norm_V
            if norm_W > 1e-12: dW /= norm_W
            
            # Adaptive learning rate based on error
            current_lr = lr * (1.0 + err)
            
            U -= current_lr * dU.reshape(n * m, rank)
            V -= current_lr * dV.reshape(m * p, rank)
            W -= current_lr * dW.reshape(p * n, rank)
            
            # Periodic exact check
            if step % 20 == 0:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
        
        # After GD, try to polish the best state found in this seed
        if best_err < 1e-3:
            U_b, V_b, W_b = best_state
            # Try Newton polish
            U_p, V_p, W_p, success = _newton_polish(U_b, V_b, W_b, n, m, p, max_iter=50, tol=1e-12)
            if success:
                ok, Us, Vs, Ws = _check_exact(U_p, V_p, W_p, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
            
            # Also check if the best state itself is close to exact
            ok, Us, Vs, Ws = _check_exact(U_b, V_b, W_b, n, m, p)
            if ok:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
                
    return None

def _newton_polish(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """
    Newton-type polish for the tensor decomposition.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    for it in range(max_iter):
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - 1.0
        err = np.sum(res**2)
        
        if err < tol:
            return U, V, W, True
        
        # Gauss-Newton step
        dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
        dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
        dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
        
        # Adaptive step size
        alpha = 1.0
        for _ in range(20):
            U_new = U - alpha * dU.reshape(n*m, rank)
            V_new = V - alpha * dV.reshape(m*p, rank)
            W_new = W - alpha * dW.reshape(p*n, rank)
            
            U_t_new = U_new.reshape(n, m, rank)
            V_t_new = V_new.reshape(m, p, rank)
            W_t_new = W_new.reshape(p, n, rank)
            
            prod_new = np.einsum('ijr,jkr,kir->ijk', U_t_new, V_t_new, W_t_new)
            res_new = prod_new - 1.0
            err_new = np.sum(res_new**2)
            
            if err_new < err:
                break
            alpha *= 0.5
        
        U = U_new
        V = V_new
        W = W_new
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
    
    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = np.sum((prod - 1.0)**2)
    return U, V, W, err < 1e-8
