"""Program for covering the sphere with 130 equal caps using differentiable min-max optimization."""
import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Use a moderately dense grid for the soft objective to balance accuracy and speed
    grid_main = generate_grid(60, 120)

    def lloyd_covering_step(pts, grid, n_iters=50, alpha=0.5):
        current = pts.copy()
        for it in range(n_iters):
            if time.time() > deadline - 15.0:
                break
            dots = np.dot(grid, current.T)
            nearest = np.argmax(dots, axis=1)
            new_pts = np.zeros_like(current)
            for i in range(N):
                mask = (nearest == i)
                if not np.any(mask):
                    new_pts[i] = current[i]
                    continue
                cell_pts = grid[mask]
                d = current[i] @ cell_pts.T
                d = np.clip(d, -1.0, 1.0)
                ang = np.arccos(d)
                temp = 3.0
                w = np.exp(ang * temp)
                w /= np.sum(w)
                new_c = np.sum(cell_pts * w[:, None], axis=0)
                nrm = np.linalg.norm(new_c)
                if nrm < 1e-10:
                    new_pts[i] = current[i]
                else:
                    new_pts[i] = new_c / nrm
            current = (1 - alpha) * current + alpha * new_pts
            current /= np.linalg.norm(current, axis=1, keepdims=True)
        return current

    def covering_radius(pts, grid=None):
        if grid is None:
            grid = grid_main
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def soft_covering_and_grad(pts, beta=50.0, grid=None):
        """
        Computes the soft covering radius and its gradient with respect to pts.
        Soft max of min distances:
        F(P) = (1/beta) log sum_y exp(beta * min_i d(y, P_i))
        
        Gradient derivation:
        Let m_y = min_i d(y, P_i).
        Let w_{y,i} be the attention weights for the min operation:
        w_{y,i} = exp(-beta * d(y, P_i)) / sum_j exp(-beta * d(y, P_j))
        
        dF/dP_k = (1/beta) * sum_y exp(beta * m_y) * (d m_y / d P_k)
        
        d m_y / d P_k = sum_i w_{y,i} * (d d(y, P_i) / d P_k)
        
        d d(y, P) / d P = - y / sqrt(1 - (y.P)^2)  (for angular distance)
        Note: The gradient of arccos(y.P) with respect to P is -y / sqrt(1 - (y.P)^2).
        """
        if grid is None:
            grid = grid_main
        
        # Compute distances
        dots = np.dot(grid, pts.T) # Shape (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots) # Shape (M, N)
        
        # Min distance for each grid point
        min_d = np.min(ang, axis=1) # Shape (M,)
        
        # Soft max over min distances
        m = np.max(min_d)
        exp_terms = np.exp(beta * (min_d - m))
        sum_exp = np.sum(exp_terms)
        F = m + np.log(sum_exp) / beta
        
        # Weights for soft max (over grid points)
        # v_y = exp(beta * m_y) / sum_j exp(beta * m_j)
        v = exp_terms / sum_exp # Shape (M,)
        
        # Weights for min (attention over points)
        # w_{y,i} = exp(-beta * d(y, P_i)) / sum_j exp(-beta * d(y, P_j))
        neg_exp = np.exp(-beta * ang) # Shape (M, N)
        sum_neg_exp = np.sum(neg_exp, axis=1, keepdims=True) # Shape (M, 1)
        w = neg_exp / sum_neg_exp # Shape (M, N)
        
        # Gradient of distance w.r.t P_k
        # grad_d_{y,i} = -grid[y] / sqrt(1 - dots[y,i]^2)
        # We need to sum this over y and i with weights v_y * w_{y,i}
        
        # To avoid division by zero, we clip the denominator
        denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-12)) # Shape (M, N)
        
        # grad_d is shape (M, N, 3)
        # grad_d[y, i, :] = -grid[y] / denom[y, i]
        
        # We want grad_P_k = sum_y v_y * sum_i w_{y,i} * grad_d_{y,i}
        # = sum_y v_y * sum_i w_{y,i} * (-grid[y] / denom[y, i])
        
        # Let A_{y,i} = v_y * w_{y,i} / denom[y,i]
        # Then grad_P_k = sum_y sum_i A_{y,i} * (-grid[y])
        
        # Combine weights:
        A = v[:, None] * w / denom # Shape (M, N)
        
        # grad_P_k = - sum_y grid[y] * (sum_i A_{y,i})
        # Let B_y = sum_i A_{y,i}
        B = np.sum(A, axis=1) # Shape (M,)
        
        # grad_P = - (grid.T @ B)
        grad_P = -(grid.T @ B) # Shape (N, 3)
        
        return F, grad_P

    def gradient_descent_optimize(pts, beta=50.0, max_iter=100, lr=0.01):
        if time.time() > deadline - 15.0:
            return pts
            
        current = pts.copy()
        prev_val = np.inf
        patience = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 15.0:
                break
                
            val, grad = soft_covering_and_grad(current, beta, grid_main)
            
            # Project gradient to tangent space
            # grad_tangent = grad - (grad . p) p
            dot_grad_p = np.sum(grad * current, axis=1, keepdims=True)
            grad_tangent = grad - dot_grad_p * current
            
            # Adaptive learning rate
            # Start small, increase if improving
            step_size = lr * (1.0 + 0.1 * np.sin(it)) # Simple modulation
            
            new_pts = current - step_size * grad_tangent
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            # Check improvement
            new_val, _ = soft_covering_and_grad(new_pts, beta, grid_main)
            
            if new_val < val:
                current = new_pts
                patience = 0
                # Optional: increase lr slightly if we've had many successes
                # lr *= 1.01
            else:
                patience += 1
                lr *= 0.9
                if patience > 10:
                    break
                    
        return current

    best_pts = None
    best_rad = np.inf

    n_restarts = 15
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if r == 0:
            pts = fibonacci_spiral(N)
        elif r == 1:
            rng = np.random.RandomState(42)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 2:
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 3:
            rng = np.random.RandomState(789)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 4:
            # Antipodal-ish start
            rng = np.random.RandomState(555)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            # Ensure spread
            pts = pts + 0.1 * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r >= 5 and best_pts is not None:
            rng = np.random.RandomState(1000 + r)
            scale = max(0.02, 0.05 - 0.005 * r)
            pts = best_pts + scale * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = fibonacci_spiral(N)
        
        # Phase 1: Lloyd to get a good initial local structure
        cand = lloyd_covering_step(pts, grid_main, n_iters=30, alpha=0.5)
        
        # Phase 2: Gradient Descent with increasing beta (annealing)
        # Start with low beta (smoother) to escape local minima, then increase
        cand = gradient_descent_optimize(cand, beta=20.0, max_iter=50, lr=0.05)
        cand = gradient_descent_optimize(cand, beta=50.0, max_iter=50, lr=0.02)
        cand = gradient_descent_optimize(cand, beta=100.0, max_iter=30, lr=0.01)
        
        rad = covering_radius(cand)
        if rad < best_rad:
            best_rad = rad
            best_pts = cand.copy()
            print(f"Restart {r}: {np.degrees(best_rad):.6f} deg")

    if best_pts is not None and time.time() < deadline - 10.0:
        # Final polish
        fp = gradient_descent_optimize(best_pts, beta=200.0, max_iter=20, lr=0.005)
        fr = covering_radius(fp)
        if fr < best_rad:
            best_rad = fr
            best_pts = fp

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
