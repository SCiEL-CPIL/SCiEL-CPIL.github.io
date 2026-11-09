import numpy as np
from scipy.optimize import minimize
import time

def _compute_entropies(P):
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    
    def proj_entropy(vals, n_bins, offset=0):
        p_flat = P.flatten()
        v_flat = (vals + offset).flatten()
        counts = np.bincount(v_flat, weights=p_flat, minlength=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        if len(probs) == 0: return 0.0
        return -np.sum(probs * np.log2(probs))
        
    H_d = proj_entropy(X - Y, 19, 9)
    H_0 = proj_entropy(X, 10, 0)
    H_1 = proj_entropy(X + Y, 19, 0)
    H_2 = proj_entropy(X + 2*Y, 28, 0)
    
    return H_d, H_0, H_1, H_2

def _compute_grads(P):
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    
    def proj_grad(vals, n_bins, offset=0):
        p_flat = P.flatten()
        v_flat = (vals + offset).flatten()
        counts = np.bincount(v_flat, weights=p_flat, minlength=n_bins)
        counts_safe = np.where(counts > 0, counts, 1.0)
        log_probs = np.log2(counts_safe)
        idx = (vals + offset)
        grad_vals = -1.0/np.log(2) * (log_probs[idx] + 1.0)
        return grad_vals

    G_d = proj_grad(X - Y, 19, 9)
    G_0 = proj_grad(X, 10, 0)
    G_1 = proj_grad(X + Y, 19, 0)
    G_2 = proj_grad(X + 2*Y, 28, 0)
    
    return G_d, G_0, G_1, G_2

def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def _objective(z):
    P = _softmax(z).reshape(10, 10)
    H_d, H_0, H_1, H_2 = _compute_entropies(P)
    M = max(H_0, H_1, H_2)
    if M < 1e-9:
        return 0.0
    return - (H_d / M)

def _gradient(z):
    P = _softmax(z).reshape(10, 10)
    H_d, H_0, H_1, H_2 = _compute_entropies(P)
    M = max(H_0, H_1, H_2)
    if M < 1e-9:
        return np.zeros(100)
        
    G_d, G_0, G_1, G_2 = _compute_grads(P)
    
    Hs = [H_0, H_1, H_2]
    Gs = [G_0, G_1, G_2]
    max_idx = np.argmax(Hs)
    G_m = Gs[max_idx]
    
    grad_P_f = (M * G_d - H_d * G_m) / (M * M)
    
    P_flat = P.flatten()
    grad_P_flat = grad_P_f.flatten()
    
    dot = np.dot(P_flat, grad_P_flat)
    grad_z = P_flat * (grad_P_flat - dot)
    
    return grad_z

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600
    
    xs = np.arange(10)
    ys = np.arange(10)
    X, Y = np.meshgrid(xs, ys)
    
    best_score = -np.inf
    best_P = None
    
    # Generate initial points using ansatz
    # P(x,y) prop to exp(u*x + v*y + w*x*y)
    # We sample u, v, w
    
    np.random.seed(42)
    
    # List of initial z vectors
    z_starts = []
    
    # Add uniform
    z_starts.append(np.zeros(100))
    
    # Add ansatz samples
    for _ in range(20):
        u = np.random.uniform(-2, 2)
        v = np.random.uniform(-2, 2)
        w = np.random.uniform(-2, 2)
        
        log_probs = u * X + v * Y + w * X * Y
        log_probs = log_probs - np.max(log_probs)
        P = np.exp(log_probs)
        P /= np.sum(P)
        
        # Convert to z
        z = np.log(P + 1e-300).flatten()
        z_starts.append(z)
        
        if time.time() > deadline - 30:
            break
            
    # Run L-BFGS-B from each start
    for i, z_init in enumerate(z_starts):
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(
                _objective, 
                z_init, 
                method='L-BFGS-B', 
                jac=_gradient,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_P = _softmax(res.x).reshape(10, 10)
        except Exception:
            pass
            
    # Fallback if no valid P found
    if best_P is None:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = np.clip(best_P, 0, None)
        s = best_P.sum()
        if s <= 1e-9:
            best_P = np.ones((10, 10)) / 100.0
        else:
            best_P /= s
            
    return best_P
