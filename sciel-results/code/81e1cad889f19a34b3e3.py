"""Structural change: use a ring-based representation with per-ring optimization,
combined with a KDTree-based force calculation for efficiency, and a multi-start
local search with ring-rotation moves."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s kill

    def compute_score(pts):
        """Compute the actual score (max common radius) for a configuration."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check all points are inside unit disk
        radii = np.sqrt(np.sum(pts**2, axis=1))
        if np.max(radii) > 1.0:
            return 0.0
        # Compute min pairwise distance
        tree = cKDTree(pts)
        # For 300 points, query_pairs is efficient
        min_dist = np.inf
        # Use query_pairs with k=1 to find nearest neighbors
        # Actually, let's just do a full pairwise for correctness, it's 300*300/2 ~ 45k pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        min_dist = np.sqrt(min_dist_sq)
        
        r_pair = min_dist / 2.0
        r_bound = 1.0 - np.max(radii)
        return min(r_pair, r_bound)

    def compute_forces(pts, r_target, k_rep=100.0, k_bound=50.0):
        """
        Compute forces on each point.
        - Repulsive force between pairs closer than 2*r_target
        - Inward force if point is too close to boundary
        Uses KDTree for neighbor search to avoid O(N^2) when possible.
        """
        forces = np.zeros_like(pts)
        radii = np.sqrt(np.sum(pts**2, axis=1))
        
        # Boundary forces
        # If |p| > 1 - r_target, push inward
        excess = np.maximum(0, radii - (1.0 - r_target))
        valid_rad = radii > 1e-12
        unit_rad = np.zeros_like(pts)
        unit_rad[valid_rad] = pts[valid_rad] / radii[valid_rad, np.newaxis]
        forces -= k_bound * excess[:, np.newaxis] * unit_rad
        
        # Pairwise repulsive forces
        # Use KDTree to find neighbors within 2*r_target * 1.5 (buffer)
        search_radius = 2.0 * r_target * 1.5
        tree = cKDTree(pts)
        pairs = tree.query_pairs(search_radius, output_type='ndarray')
        
        if len(pairs) > 0:
            for i, j in pairs:
                if i == j:
                    continue
                d_vec = pts[i] - pts[j]
                d = np.sqrt(np.sum(d_vec**2))
                if d < 1e-12:
                    # Random small perturbation
                    d_vec = np.random.normal(0, 1e-6, 2)
                    d = np.sqrt(np.sum(d_vec**2))
                if d < 2.0 * r_target:
                    # Repulsive force: strength proportional to overlap
                    overlap = 2.0 * r_target - d
                    f_mag = k_rep * overlap
                    d_unit = d_vec / d
                    forces[i] += f_mag * d_unit
                    forces[j] -= f_mag * d_unit
        
        return forces

    def local_search(pts, r_target, max_iters=2000, lr_init=0.005, lr_decay=0.999, T_init=0.002, T_final=1e-5):
        """
        Local search with simulated annealing on a fixed r_target.
        Tries to find a configuration where min_radius >= r_target.
        Returns best pts found and its actual min_radius.
        """
        current_pts = pts.copy()
        best_pts = current_pts.copy()
        best_r = compute_score(best_pts)
        
        T = T_init
        lr = lr_init
        tree = cKDTree(current_pts)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            forces = compute_forces(current_pts, r_target)
            
            # Add noise
            noise = np.random.normal(0, T, size=current_pts.shape)
            new_pts = current_pts + lr * forces + noise
            
            # Clamp to stay within a safe region (slightly inside unit disk)
            new_radii = np.sqrt(np.sum(new_pts**2, axis=1))
            max_safe = 1.0 - r_target * 0.5  # Allow some boundary flexibility
            too_far = new_radii > max_safe
            if np.any(too_far):
                new_pts[too_far] *= max_safe / new_radii[too_far, np.newaxis]
            
            # Quick score check: compute min radius
            new_r = compute_score(new_pts)
            
            # Acceptance: always accept if better, otherwise Metropolis
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
            
            # For SA, we use energy-based acceptance
            # Energy: sum of overlaps for r_target
            # Simplified: use the score directly. If new_r >= r_target, energy is low.
            # Let's define energy = max(0, r_target - new_r) + small penalty for boundary
            curr_energy = max(0, r_target - compute_score(current_pts))
            new_energy = max(0, r_target - new_r)
            
            if new_energy < curr_energy:
                current_pts = new_pts
            else:
                delta_e = new_energy - curr_energy
                if delta_e > 0:
                    if np.random.random() < np.exp(-delta_e / max(T, 1e-10)):
                        current_pts = new_pts
                else:
                    current_pts = new_pts
            
            T *= 0.9995
            if T < T_final:
                T = T_final
            lr *= lr_decay
            if lr < 1e-6:
                lr = 1e-6
        
        return best_pts, best_r

    def ring_initialization(num_rings):
        """
        Generate points on concentric rings.
        Ring k (1-indexed) has radius r_k and n_k points.
        We choose n_k proportional to r_k (to maintain roughly equal spacing).
        """
        pts = []
        # Total N = 300
        # Let's use a simple scheme: ring k has roughly k * 10 points? 
        # Or distribute evenly: each ring gets N/num_rings points.
        # Better: radius r_k = k / num_rings * (1 - r_est)
        # n_k = round(N * (r_k^2 - r_{k-1}^2) / (1 - r_est)^2) ? 
        # Simpler: just divide N evenly among rings.
        
        r_est = 0.053
        max_r = 1.0 - r_est
        
        for k in range(1, num_rings + 1):
            r_k = k / num_rings * max_r
            n_k = N // num_rings
            if k == num_rings:
                n_k = N - (num_rings - 1) * (N // num_rings)
            
            theta = np.linspace(0, 2 * np.pi, n_k, endpoint=False)
            # Add a small random offset to break symmetry
            theta += np.random.uniform(-0.1, 0.1, n_k)
            x = r_k * np.cos(theta)
            y = r_k * np.sin(theta)
            pts.append(np.column_stack((x, y)))
        
        pts = np.vstack(pts)
        return pts

    def hex_initialization():
        """Generate a hex lattice initialization."""
        r_est = 0.05
        spacing = 2.0 * r_est
        k_max = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - r_est:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) >= N:
            # Keep the N points closest to the center
            dists = np.sqrt(np.sum(pts**2, axis=1))
            idx = np.argsort(dists, kind='stable')[:N]
            pts = pts[idx]
        else:
            # Pad with random points if needed
            remaining = N - len(pts)
            r = np.sqrt(np.random.random(remaining)) * (1.0 - r_est)
            theta = np.random.random(remaining) * 2 * np.pi
            pad = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
            pts = np.vstack([pts, pad])
        return pts

    def random_initialization():
        """Random uniform in disk."""
        r_est = 0.05
        r = np.sqrt(np.random.random(N)) * (1.0 - r_est)
        theta = np.random.random(N) * 2 * np.pi
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))

    # Generate initial configurations
    inits = []
    
    # 1. Hex lattice
    inits.append(hex_initialization())
    
    # 2. Ring-based with different numbers of rings
    for num_rings in [10, 15, 20]:
        inits.append(ring_initialization(num_rings))
    
    # 3. Random
    for _ in range(3):
        inits.append(random_initialization())
    
    # Evaluate initial configurations
    best_pts = None
    best_r = 0.0
    for init in inits:
        r = compute_score(init)
        if r > best_r:
            best_r = r
            best_pts = init.copy()
    
    if best_pts is None:
        best_pts = hex_initialization()
        best_r = compute_score(best_pts)
    
    # Multi-start local search with increasing r_target
    # Start from a conservative r_target and increase
    r_target = best_r + 0.001
    max_r_target = 0.054
    
    num_restarts = 8
    time_per_restart = (deadline - time.time() - 10) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if time_per_restart < 5:
            break
        
        # Choose initialization
        if restart == 0:
            start_pts = best_pts.copy()
        else:
            # Use a different init
            start_pts = inits[restart % len(inits)].copy()
            # If this init is worse than best, blend with best?
            # Or just use it as is.
        
        # Set r_target for this restart
        if restart == 0:
            r_target = best_r + 0.0005
        else:
            # For other restarts, try to reach a higher r_target
            r_target = min(best_r + 0.001, max_r_target)
        
        # Run local search
        t_start = time.time()
        max_iters = int(time_per_restart * 100)  # Rough estimate
        max_iters = max(100, min(max_iters, 5000))
        
        new_pts, new_r = local_search(start_pts, r_target, max_iters=max_iters, 
                                       lr_init=0.003, lr_decay=0.9995, 
                                       T_init=0.001, T_final=1e-6)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            # Update r_target for next restart
            r_target = best_r + 0.0005
    
    # Final polish: try to squeeze out a bit more
    if time.time() < deadline - 5:
        r_target = best_r + 0.0002
        new_pts, new_r = local_search(best_pts, r_target, max_iters=2000,
                                       lr_init=0.001, lr_decay=0.9998,
                                       T_init=0.0005, T_final=1e-7)
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
    
    # Final validation
    final_r = compute_score(best_pts)
    if final_r <= 0:
        # Fallback to hex lattice
        best_pts = hex_initialization()
    
    return best_pts

# EVOLVE-BLOCK-END
