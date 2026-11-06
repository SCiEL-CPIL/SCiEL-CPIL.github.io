import numpy as np
import time
import sys
from scipy.spatial import ConvexHull

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius in radians.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_circumradius = 0.0
        for i, j, k in facets:
            p_i = points[i]
            p_j = points[j]
            p_k = points[k]
            
            # Circumcenter direction
            c = p_i + p_j + p_k
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                continue
            c = c / norm_c
            
            # Circumradius
            dot = np.dot(c, p_i)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)
            
            if theta > max_circumradius:
                max_circumradius = theta
                
        return max_circumradius
    except Exception:
        return np.pi

def _fibonacci_sphere(n):
    """Generate a Fibonacci sphere distribution."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def construct():
    """
    Constructs a set of N points on the unit sphere using Parallel Tempering.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s kill
    
    # Parameters
    n_replicas = 10
    n_steps = 5000  # Total steps per replica chain (approx)
    exchange_interval = 20 # Steps between exchange attempts
    
    # Temperatures: Geometric progression
    # T_low should be very small for fine tuning, T_high large enough for exploration
    T_low = 1e-5
    T_high = 0.05
    temps = np.geomspace(T_low, T_high, n_replicas)
    
    rng = np.random.default_rng(42)
    
    # Initialize replicas
    # Replica 0 starts with Fibonacci, others get random perturbations or different seeds
    points_list = []
    for r in range(n_replicas):
        if r == 0:
            pts = _fibonacci_sphere(N)
        else:
            pts = _fibonacci_sphere(N)
            # Add some noise to diversify initial states
            noise = rng.normal(0, 0.05 * (r + 1), (N, 3))
            pts = pts + noise
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
        points_list.append(pts)
    
    # Compute initial radii
    radii = [_compute_covering_radius(pts) for pts in points_list]
    
    # Track best global solution
    best_radius = min(radii)
    best_points = points_list[np.argmin(radii)].copy()
    
    # Helper to get worst facet and its vertices
    def get_worst_facet_info(pts):
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            max_cr = 0.0
            worst_idx = -1
            for idx, (i, j, k) in enumerate(facets):
                p_i = pts[i]
                p_j = pts[j]
                p_k = pts[k]
                c = p_i + p_j + p_k
                norm_c = np.linalg.norm(c)
                if norm_c < 1e-10:
                    continue
                c = c / norm_c
                dot = np.clip(np.dot(c, p_i), -1.0, 1.0)
                theta = np.arccos(dot)
                if theta > max_cr:
                    max_cr = theta
                    worst_idx = idx
            if worst_idx == -1:
                return None, max_cr
            i, j, k = facets[worst_idx]
            return [i, j, k], max_cr
        except Exception:
            return None, np.pi

    # Main Loop
    step_count = 0
    while time.time() - start_time < time_limit:
        for r in range(n_replicas):
            if time.time() - start_time > time_limit:
                break
            
            T = temps[r]
            pts = points_list[r]
            current_radius = radii[r]
            
            # 1. Propose a move
            # Identify worst facet
            worst_vertices, _ = get_worst_facet_info(pts)
            
            if worst_vertices is None:
                # Fallback: random move
                v_idx = rng.integers(0, N)
                p_v = pts[v_idx]
                noise = rng.normal(0, 0.05, 3)
                new_p_v = p_v + noise
                new_p_v = new_p_v / np.linalg.norm(new_p_v)
                old_p_v = p_v.copy()
                pts[v_idx] = new_p_v
            else:
                # Move one of the worst vertices towards the hole center
                v_idx = rng.choice(worst_vertices)
                p_v = pts[v_idx]
                
                # Compute center of the worst facet
                i, j, k = worst_vertices
                c = pts[i] + pts[j] + pts[k]
                norm_c = np.linalg.norm(c)
                if norm_c < 1e-10:
                    c = p_v
                else:
                    c = c / norm_c
                    
                # Direction towards center, projected to tangent
                dir_vec = c - p_v
                dot_proj = np.dot(dir_vec, p_v)
                dir_tangent = dir_vec - dot_proj * p_v
                norm_dir = np.linalg.norm(dir_tangent)
                
                if norm_dir > 1e-10:
                    dir_tangent = dir_tangent / norm_dir
                    # Step size scales with temperature
                    step_size = 0.01 * (1.0 + 10.0 * T)
                    noise = rng.normal(0, step_size * 0.5, 3)
                    new_p_v = p_v + step_size * dir_tangent + noise
                else:
                    new_p_v = p_v + rng.normal(0, 0.01, 3)
                
                new_p_v = new_p_v / np.linalg.norm(new_p_v)
                old_p_v = p_v.copy()
                pts[v_idx] = new_p_v
            
            # 2. Evaluate
            new_radius = _compute_covering_radius(pts)
            
            # 3. Metropolis Acceptance
            if new_radius < current_radius:
                accept = True
            else:
                delta = new_radius - current_radius
                # Use temperature T for this replica
                prob = np.exp(-delta / T)
                accept = rng.random() < prob
            
            if accept:
                radii[r] = new_radius
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = pts.copy()
            else:
                # Revert
                if worst_vertices is not None:
                    pts[v_idx] = old_p_v
                else:
                    pts[v_idx] = old_p_v

        # 4. Exchange Attempts (Parallel Tempering)
        step_count += 1
        if step_count % exchange_interval == 0:
            # Try to exchange adjacent replicas
            # Usually exchange from high T to low T or vice versa
            # We can try all adjacent pairs or just one random pair
            # Let's try all adjacent pairs to be thorough, but it's O(N_replicas)
            for r in range(n_replicas - 1):
                r1 = r
                r2 = r + 1
                T1 = temps[r1]
                T2 = temps[r2]
                E1 = radii[r1]
                E2 = radii[r2]
                
                # Delta E = (E2 - E1) * (1/T1 - 1/T2)
                # Accept with prob 1 if Delta E < 0, else exp(-Delta E)
                delta_E = (E2 - E1) * (1.0/T1 - 1.0/T2)
                
                if delta_E < 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta_E)
                
                if accept:
                    # Swap points and radii
                    points_list[r1], points_list[r2] = points_list[r2], points_list[r1]
                    radii[r1], radii[r2] = radii[r2], radii[r1]

    # Final cleanup
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    print(f"Covering Radius: {_compute_covering_radius(pts) * 180 / np.pi:.6f} degrees")
