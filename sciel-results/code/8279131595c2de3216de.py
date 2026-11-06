"""
Program for packing 50 equal circles in a unit circle.
Strategy: Discrete combinatorial search over ring configurations (integer counts per ring),
with continuous optimization of ring radii and angular offsets via gradient descent.
This reduces the search space by enforcing a concentric ring structure.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50
DEADLINE = time.time() + 140.0  # Safety margin out of 150s

def compute_radius(points):
    """Compute the packing radius for given centres."""
    points = np.asarray(points, dtype=np.float64)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_dists.min(), min_pair_dist / 2.0)

def fast_radius(points):
    """Fast radius computation."""
    points = np.asarray(points, dtype=np.float64)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    return min(boundary_dists.min(), min_pair_dist / 2.0)

def generate_ring_config(total_circles):
    """
    Generate a random ring configuration: list of (count, ring_index).
    Returns a list of counts per ring, e.g., [1, 6, 12, 18, 13] for 50 circles.
    """
    configs = []
    # Try various numbers of rings
    for n_rings in range(2, 8):
        # Distribute circles among rings
        # Use a heuristic: more circles in outer rings
        weights = np.linspace(1, 3, n_rings)
        weights /= weights.sum()
        counts = np.round(weights * total_circles).astype(int)
        # Adjust to ensure sum is total_circles
        diff = total_circles - counts.sum()
        if diff > 0:
            for _ in range(diff):
                idx = np.random.choice(n_rings)
                counts[idx] += 1
        elif diff < 0:
            for _ in range(-diff):
                # Remove from the largest count
                idx = np.argmax(counts)
                if counts[idx] > 0:
                    counts[idx] -= 1
        # Ensure first ring has at least 1 if n_rings > 1
        if counts[0] == 0 and n_rings > 1:
            counts[0] = 1
            counts[-1] -= 1
        if counts.sum() == total_circles and all(c >= 0 for c in counts):
            configs.append(counts.tolist())
    
    if not configs:
        # Fallback: 1 center, 6, 12, 18, 13
        return [1, 6, 12, 18, 13]
    
    return random.choice(configs)

def build_points_from_rings(counts, radii, angles):
    """
    Build the N points from ring configuration.
    counts: list of int, number of circles per ring
    radii: list of float, radius of each ring
    angles: list of float, angular offset for each ring
    Returns array of shape (N, 2).
    """
    points = []
    for ring_idx, (count, radius, offset) in enumerate(zip(counts, radii, angles)):
        if count == 0:
            continue
        if radius < 1e-6:
            # Center point
            points.append([0.0, 0.0])
        else:
            for k in range(count):
                angle = offset + 2 * np.pi * k / count
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                points.append([x, y])
    
    return np.array(points)

def ring_objective(params, counts):
    """
    Objective function for ring optimization.
    params: [r_0, r_1, ..., r_{m-1}, a_0, a_1, ..., a_{m-1}]
    where r_i is the radius of ring i, a_i is the angular offset.
    Returns -radius (to be minimized).
    """
    n_rings = len(counts)
    radii = params[:n_rings]
    angles = params[n_rings:]
    
    # Ensure radii are non-negative and ordered
    radii = np.abs(radii)
    # Sort radii and corresponding angles? No, the order matters for the ring index.
    # But we can enforce r_0 < r_1 < ... < r_{m-1} by sorting?
    # Actually, the physical meaning is ring 0 is innermost.
    # Let's just use the values as is, but clip to [0, 1].
    radii = np.clip(radii, 0.0, 0.99)
    
    points = build_points_from_rings(counts, radii, angles)
    
    if len(points) != N:
        return 1.0  # Invalid configuration, large penalty
    
    r = fast_radius(points)
    return -r

def optimize_ring_config(counts, n_iter=1000):
    """
    Optimize the radii and angles for a given ring configuration.
    """
    n_rings = len(counts)
    
    # Initial guess for radii
    # Spread radii roughly evenly from 0 to 0.9
    initial_radii = np.linspace(0.0, 0.9, n_rings)
    # Initial angles: random
    initial_angles = np.random.uniform(0, 2*np.pi, n_rings)
    
    x0 = np.concatenate([initial_radii, initial_angles])
    
    # Bounds
    bounds = [(0.0, 0.99)] * n_rings + [(0.0, 2*np.pi)] * n_rings
    
    # Use Nelder-Mead or L-BFGS-B?
    # Nelder-Mead is derivative-free and robust.
    # L-BFGS-B requires gradients, which we don't have easily.
    # Let's try Nelder-Mead first.
    
    result = minimize(
        ring_objective, 
        x0, 
        args=(counts,), 
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': n_iter, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    
    if result.success:
        radii = result.x[:n_rings]
        angles = result.x[n_rings:]
        radii = np.clip(radii, 0.0, 0.99)
        points = build_points_from_rings(counts, radii, angles)
        if len(points) == N:
            r = fast_radius(points)
            return points, r
    else:
        # Fallback to initial guess
        points = build_points_from_rings(counts, initial_radii, initial_angles)
        if len(points) == N:
            r = fast_radius(points)
            return points, r
    
    return None, 0.0

def local_polish(points, n_steps=200, step_size=0.005):
    """
    Simple coordinate descent polish to fix minor overlaps or boundary violations
    and squeeze out the last bit of radius.
    """
    x = points.copy()
    current_r = fast_radius(x)
    
    for _ in range(n_steps):
        if time.time() > DEADLINE - 2.0:
            break
        improved = False
        indices = np.random.permutation(N)
        for i in indices:
            best_x_i = x[i].copy()
            best_r_i = current_r
            
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x_i = x[i] + np.array([dx, dy])
                    if np.linalg.norm(new_x_i) >= 1.0:
                        continue
                    x[i] = new_x_i
                    new_r = fast_radius(x)
                    if new_r > best_r_i:
                        best_r_i = new_r
                        best_x_i = new_x_i.copy()
                    else:
                        x[i] = best_x_i # revert
            if best_r_i > current_r:
                x[i] = best_x_i
                current_r = best_r_i
                improved = True
        if not improved:
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    return x, current_r

def construct():
    global DEADLINE
    DEADLINE = time.time() + 140.0
    
    np.random.seed(42)
    random.seed(42)
    
    best_points = None
    best_radius = 0.0
    
    # Generate a variety of ring configurations
    configs = []
    
    # Specific known good configs for 50 circles
    # 1, 6, 12, 18, 13
    configs.append([1, 6, 12, 18, 13])
    # 1, 6, 12, 12, 12, 7
    configs.append([1, 6, 12, 12, 12, 7])
    # 1, 6, 12, 18, 10, 3
    configs.append([1, 6, 12, 18, 10, 3])
    # 1, 6, 12, 14, 14, 3
    configs.append([1, 6, 12, 14, 14, 3])
    
    # Generate random configs
    for _ in range(20):
        c = generate_ring_config(N)
        if c not in configs:
            configs.append(c)
    
    # Shuffle configs
    random.shuffle(configs)
    
    # Time budget per config
    time_per_config = (DEADLINE - time.time() - 10.0) / len(configs)
    
    for counts in configs:
        if time.time() > DEADLINE - 10.0:
            break
        
        # Adjust n_iter based on time
        n_iter = int(min(500, max(50, time_per_config * 50)))
        
        points, r = optimize_ring_config(counts, n_iter=n_iter)
        
        if points is not None and r > best_radius:
            best_radius = r
            best_points = points.copy()
            
    # Final polish
    if best_points is not None and time.time() < DEADLINE - 5.0:
        polished_points, polished_r = local_polish(best_points)
        if polished_r > best_radius:
            best_points = polished_points
            best_radius = polished_r
            
    # Fallback if no valid points found
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback to simple hexagonal
        spacing = 0.2
        points_list = []
        for j in range(-5, 6):
            for k in range(-5, 6):
                px = k * spacing + (0.5 * spacing if j % 2 else 0)
                py = j * spacing * np.sqrt(3) / 2
                if np.sqrt(px * px + py * py) < 0.95:
                    points_list.append([px, py])
        if len(points_list) >= N:
            dists = np.linalg.norm(np.array(points_list), axis=1)
            indices = np.argsort(dists)
            best_points = np.array(points_list)[indices[:N]]
        else:
            best_points = np.random.uniform(-0.9, 0.9, size=(N, 2))
            
    # Final validation and clipping
    best_points = np.asarray(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] >= 1.0:
            best_points[i] *= 0.999 / norms[i]
            
    return best_points

if __name__ == "__main__":
    points = construct()
    print("Final radius:", compute_radius(points))
