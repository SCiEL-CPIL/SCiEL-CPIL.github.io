import numpy as np
from scipy.special import ellipk

N = 26

def construct():
    k = 1.0 / np.sqrt(2.0)
    K_val = ellipk(k**2)
    scale = np.sqrt(2.0) * K_val
    
    def compute_radii(positions):
        xs = positions[:N]
        ys = positions[N:]
        rs = np.zeros(N)
        
        # Wall constraints
        for i in range(N):
            rs[i] = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
            
        # Neighbor constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                val = dist / 2.0
                if val < rs[i]:
                    rs[i] = val
                if val < rs[j]:
                    rs[j] = val
        return rs

    def objective(positions):
        rs = compute_radii(positions)
        return -np.sum(rs)

    def disk_to_square(theta):
        """Map angle theta in disk to point in square using SC map"""
        from scipy.integrate import quad
        # F(theta, k) = integral from 0 to theta of dt/sqrt(1-k^2*sin^2(t))
        def integrand(t):
            return 1.0 / np.sqrt(1.0 - k**2 * np.sin(t)**2)
        F_theta, _ = quad(integrand, 0, theta)
        x = scale * F_theta + 0.5
        y = 0.5  # This only gives the real axis; need full 2D map
        return x, y

    def better_init():
        # Use SC map to generate well-distributed points
        thetas = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.uniform(-0.1, 0.1, N)
        np.random.shuffle(thetas)
        
        pos = np.zeros(2*N)
        for i, theta in enumerate(thetas):
            # Use a radial distribution
            r = np.sqrt(np.random.uniform(0, 1))
            x = 0.5 + r * np.cos(theta) * 0.45
            y = 0.5 + r * np.sin(theta) * 0.45
            pos[i] = np.clip(x, 0.05, 0.95)
            pos[N+i] = np.clip(y, 0.05, 0.95)
        return pos

    best_pos = None
    best_score = -np.inf
    np.random.seed(42)
    
    for restart in range(10):
        if restart == 0:
            current_pos = better_init()
        else:
            current_pos = np.random.rand(2*N)
            current_pos = np.clip(current_pos, 0.08, 0.92)
            
        current_obj = objective(current_pos)
        
        T = 0.04
        T_min = 0.0003
        alpha = 0.996
        steps = 6000
        
        for step in range(steps):
            new_pos = current_pos.copy()
            n_perturb = np.random.randint(1, 4)
            indices = np.random.choice(2*N, n_perturb, replace=False)
            for idx in indices:
                delta = np.random.normal(0, T)
                new_pos[idx] += delta
                new_pos[idx] = np.clip(new_pos[idx], 0.0, 1.0)
            
            new_obj = objective(new_pos)
            delta_obj = new_obj - current_obj
            
            if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / T):
                current_pos = new_pos
                current_obj = new_obj
                
            T *= alpha
            if T < T_min:
                T = T_min
                
        # Local refinement
        for _ in range(300):
            for i in range(2*N):
                for delta in [-0.005, -0.002, -0.0005, 0.0005, 0.002, 0.005]:
                    trial = current_pos.copy()
                    trial[i] = np.clip(trial[i] + delta, 0.0, 1.0)
                    if objective(trial) < objective(current_pos):
                        current_pos = trial
                        break
        
        rs = compute_radii(current_pos)
        score = np.sum(rs)
        
        if score > best_score:
            best_score = score
            best_pos = current_pos.copy()

    xs = best_pos[:N]
    ys = best_pos[N:]
    rs = compute_radii(best_pos)
    
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = rs
    
    return result
