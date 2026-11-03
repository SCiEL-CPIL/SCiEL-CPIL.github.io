import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    np.random.seed(42)
    
    def fibonacci_spiral(offset=0.5):
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        pts = np.column_stack((x, y, z))
        return pts

    def log_energy(pts):
        # pts: (N, 3) normalized
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        d = np.clip(d, 1e-8, None)
        return -np.sum(np.log(d))

    def log_energy_grad(pts):
        # Returns gradient of energy w.r.t pts, assuming pts are on sphere
        # grad[i] = sum_{j!=i} (pts[j] - pts[i]) / d_ij^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists_safe = np.where(dists > 1e-8, dists, 1e-8)
        
        # w[i,j] = 1/dists[i,j]^2
        w = np.zeros((N, N))
        iu = np.triu_indices(N, k=1)
        w[iu] = 1.0 / (dists_safe[iu] ** 2)
        w = w + w.T
        
        # grad_pts[i] = sum_j w[i,j] * (pts[j] - pts[i])
        # = (w @ pts).T - (w.sum(axis=1) * pts).T
        grad_pts = (w @ pts).T - (w.sum(axis=1, keepdims=True) * pts)
        
        # Project to tangent space: grad_tan = grad - (grad . pts) * pts
        dot = np.sum(grad_pts * pts, axis=1, keepdims=True)
        grad_tan = grad_pts - dot * pts
        
        return grad_tan

    def riemannian_optimize(pts0, steps=2000, lr=0.01, decay=0.999):
        pts = pts0.copy()
        m = np.zeros_like(pts)
        v = np.zeros_like(pts)
        t = 0
        best_pts = pts.copy()
        best_e = log_energy(pts)
        
        for i in range(steps):
            t += 1
            e = log_energy(pts)
            if e < best_e:
                best_e = e
                best_pts = pts.copy()
                
            grad = log_energy_grad(pts)
            
            # Adam update
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad ** 2)
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            
            current_lr = lr * (0.5 ** (i // 1000)) # Simple decay
            pts = pts - current_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # Retraction: Normalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pts = pts / norms
            
        return best_pts

    inits = []
    inits.append(fibonacci_spiral())
    for _ in range(5):
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        inits.append(pts)
        
    best_energy = np.inf
    best_pts = None
    
    for init in inits:
        # Run Riemannian Adam
        res_pts = riemannian_optimize(init, steps=1000, lr=0.05)
        
        # Final polish with L-BFGS-B on Cartesian?
        # Since we are on sphere, we can just do a few more Riemannian steps with smaller LR
        res_pts = riemannian_optimize(res_pts, steps=500, lr=0.01)
        
        e = log_energy(res_pts)
        if e < best_energy:
            best_energy = e
            best_pts = res_pts

    return best_pts
