import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def compute_radius(points):
    """Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0


def energy_and_grad(x, r_target):
    """
    Energy function for simulated annealing / gradient optimization.
    E = sum of overlap penalties + boundary violation penalties.
    """
    pts = x.reshape(N, 2)
    
    # Boundary penalty: (max(0, |c_i| - (1 - r_target)))^2
    norms = np.linalg.norm(pts, axis=1)
    boundary_viol = np.maximum(0, norms - (1.0 - r_target))
    E_boundary = np.sum(boundary_viol ** 2)
    
    # Pairwise penalty: (max(0, 2*r_target - d_ij))^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    overlap = np.maximum(0, 2.0 * r_target - dists)
    E_overlap = np.sum(overlap ** 2)
    
    E = E_boundary + E_overlap
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Boundary gradient
    for i in range(N):
        if boundary_viol[i] > 0 and norms[i] > 1e-10:
            grad[2*i:2*i+2] += 2 * boundary_viol[i] * pts[i] / norms[i]
    
    # Pairwise gradient
    for i in range(N):
        for j in range(i + 1, N):
            d = dists[i, j]
            if d < 2.0 * r_target and d > 1e-10:
                vec = pts[i] - pts[j]
                penalty = 2.0 * (2.0 * r_target - d)
                grad[2*i:2*i+2] += penalty * vec / d
                grad[2*j:2*j+2] += -penalty * vec / d
    
    return E, grad


def simulated_annealing(pts, r_target, time_limit=10.0):
    """
    Simulated annealing to find a good configuration.
    """
    pts = pts.copy()
    start_time = time.time()
    T = 0.05  # initial temperature
    T_min = 0.001
    cooling = 0.999
    
    best_pts = pts.copy()
    best_E = energy_and_grad(pts.flatten(), r_target)[0]
    
    while time.time() - start_time < time_limit:
        # Pick a random particle
        i = np.random.randint(N)
        
        # Random displacement
        disp = np.random.normal(0, T, 2)
        new_pts = pts.copy()
        new_pts[i] += disp
        
        # Ensure inside unit circle (soft constraint)
        norm = np.linalg.norm(new_pts[i])
        if norm > 1.0:
            new_pts[i] = new_pts[i] / norm * (1.0 - 0.01)
        
        E_old = energy_and_grad(pts.flatten(), r_target)[0]
        E_new = energy_and_grad(new_pts.flatten(), r_target)[0]
        
        delta_E = E_new - E_old
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / max(T, T_min)):
            pts = new_pts
            if E_new < best_E:
                best_E = E_new
                best_pts = new_pts.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts


def gradient_optimize(pts, r_target, time_limit=5.0):
    """
    Use gradient descent to polish the configuration.
    """
    x = pts.flatten()
    start_time = time.time()
    lr = 0.01
    
    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        E, grad = energy_and_grad(x, r_target)
        x -= lr * grad
        
        # Reshape and project inside unit circle
        pts_tmp = x.reshape(N, 2)
        norms = np.linalg.norm(pts_tmp, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                pts_tmp[i] = pts_tmp[i] / norms[i] * (1.0 - 0.01)
        x = pts_tmp.flatten()
        
        # Adaptive learning rate
        if E < 1e-6:
            lr *= 0.9
    
    return x.reshape(N, 2)


def initial_hexagonal():
    """Generate initial points using a hexagonal lattice pattern."""
    spacing = 0.25
    points = []
    
    max_k = 8
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    
    if len(points) >= N:
        selected = points[indices[:N]]
    else:
        needed = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, size=(needed, 2))
        selected = np.vstack([points, random_pts])
        
    return selected


def initial_rings():
    """Generate initial points using concentric rings."""
    points = np.zeros((N, 2))
    points[0] = [0, 0]
    count = 1
    
    ring_radius = 0.2
    ring_step = 0.2
    
    while count < N:
        n_in_ring = int(2 * np.pi * ring_radius / 0.25)
        if n_in_ring < 1:
            n_in_ring = 1
        
        for k in range(n_in_ring):
            if count >= N:
                break
            angle = 2 * np.pi * k / n_in_ring
            points[count, 0] = ring_radius * np.cos(angle)
            points[count, 1] = ring_radius * np.sin(angle)
            count += 1
        
        ring_radius += ring_step
    
    return points


def construct():
    deadline = time.time() + DEADLINE
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # Try multiple initializations and optimization strategies
    initializations = [
        initial_hexagonal(),
        initial_rings()
    ]
    
    # Add a random initialization
    random_pts = np.random.uniform(-0.9, 0.9, size=(N, 2))
    # Project inside unit circle
    norms = np.linalg.norm(random_pts, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            random_pts[i] = random_pts[i] / norms[i] * 0.9
    initializations.append(random_pts)
    
    for init_pts in initializations:
        if time.time() > deadline - 10:
            break
        
        # Start with a reasonable radius estimate
        r_est = compute_radius(init_pts)
        if r_est <= 0:
            r_est = 0.1
        
        # Simulated annealing
        try:
            sa_pts = simulated_annealing(init_pts, r_est, time_limit=15.0)
            r_sa = compute_radius(sa_pts)
            
            if r_sa > best_overall_r:
                best_overall_r = r_sa
                best_overall_points = sa_pts.copy()
        except Exception:
            pass
        
        # Gradient optimization from the SA result
        if best_overall_points is not None and time.time() < deadline - 5:
            try:
                grad_pts = gradient_optimize(best_overall_points, best_overall_r, time_limit=5.0)
                r_grad = compute_radius(grad_pts)
                
                if r_grad > best_overall_r:
                    best_overall_r = r_grad
                    best_overall_points = grad_pts.copy()
            except Exception:
                pass
    
    # Final polish: try to increase radius slightly
    if best_overall_points is not None and time.time() < deadline - 5:
        try:
            # Try to increase radius by small increments
            for _ in range(10):
                if time.time() > deadline - 2:
                    break
                r_try = best_overall_r * 1.001
                polished = gradient_optimize(best_overall_points, r_try, time_limit=2.0)
                r_new = compute_radius(polished)
                if r_new > best_overall_r:
                    best_overall_r = r_new
                    best_overall_points = polished.copy()
                else:
                    break
        except Exception:
            pass
    
    return best_overall_points if best_overall_points is not None else initial_hexagonal()


if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
