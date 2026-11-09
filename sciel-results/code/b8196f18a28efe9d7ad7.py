import numpy as np

def construct():
    """
    Constructs the centers of 300 circles inside a unit circle.
    Uses Simulated Annealing with Poisson Disk Sampling initialization,
    followed by a short Local Search polish.
    """
    
    def poisson_disk_sampling(n_points, domain_radius=1.0, min_dist=0.09, max_attempts=1000):
        points = []
        # Start with a random point in the center
        start = np.random.uniform(-domain_radius*0.5, domain_radius*0.5, 2)
        if np.linalg.norm(start) > domain_radius:
            start = np.array([0.0, 0.0])
        points.append(start)
        
        active = [start]
        
        while len(points) < n_points and active:
            idx = np.random.randint(len(active))
            p = active[idx]
            
            success = False
            for _ in range(max_attempts):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = min_dist * np.sqrt(np.random.uniform(1.0, 4.0))
                new_p = p + radius * np.array([np.cos(angle), np.sin(angle)])
                
                if np.linalg.norm(new_p) > domain_radius:
                    continue
                
                dists = np.linalg.norm(np.array(points) - new_p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(new_p)
                    active.append(new_p)
                    success = True
                    break
            
            if not success:
                active.pop(idx)
                
        while len(points) < n_points:
            p = np.random.uniform(-domain_radius, domain_radius, 2)
            if np.linalg.norm(p) <= domain_radius:
                points.append(p)
                
        return np.array(points[:n_points])

    def get_radius(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dists_sq))
        
        radii = np.linalg.norm(p, axis=1)
        min_boundary_dist = 1.0 - np.min(radii)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    # 1. Init
    pos = poisson_disk_sampling(N, 1.0, 0.09)
    
    # Ensure inside
    radii = np.linalg.norm(pos, axis=1)
    outside = radii > 1.0
    if np.any(outside):
        pos[outside] = pos[outside] / radii[outside, np.newaxis] * 0.99

    r_current = get_radius(pos)
    
    # 2. Simulated Annealing
    T_start = 0.02
    T_end = 1e-6
    steps = 15000
    cooling_factor = (T_end / T_start) ** (1.0 / steps)
    
    best_pos = pos.copy()
    best_r = r_current
    
    for step in range(steps):
        T = T_start * (cooling_factor ** step)
        
        i = np.random.randint(N)
        
        step_size = T * 0.5
        delta = np.random.normal(0, step_size, 2)
        
        new_pos = pos.copy()
        new_pos[i] = pos[i] + delta
        
        r_new_i = np.linalg.norm(new_pos[i])
        if r_new_i > 1.0:
            new_pos[i] = new_pos[i] / r_new_i
            
        r_new = get_radius(new_pos)
        
        if r_new > r_current:
            pos = new_pos
            r_current = r_new
            if r_current > best_r:
                best_r = r_current
                best_pos = pos.copy()
        else:
            delta_r = r_new - r_current
            if delta_r == 0:
                accept = True
            else:
                prob = np.exp(delta_r / T)
                accept = np.random.random() < prob
            
            if accept:
                pos = new_pos
                r_current = r_new
                if r_current > best_r:
                    best_r = r_current
                    best_pos = pos.copy()

    # 3. Local Search Polish
    # Simple gradient-like push
    max_iter = 500
    step_size_ls = 0.002
    
    for _ in range(max_iter):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist_sq = np.min(dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        radii = np.linalg.norm(pos, axis=1)
        boundary_dists = 1.0 - radii
        min_boundary_dist = np.min(boundary_dists)
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if min_pair_dist / 2.0 <= min_boundary_dist:
            threshold_sq = (min_pair_dist * 1.05)**2
            close_mask = dists_sq < threshold_sq
            np.fill_diagonal(close_mask, False)
            
            sum_vecs = np.sum(
                np.where(close_mask[:, :, np.newaxis], diff, 0.0), 
                axis=1
            )
            forces = sum_vecs
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            forces = forces / norms
            
            pos += forces * step_size_ls
        else:
            boundary_mask = boundary_dists < (min_boundary_dist + 1e-4)
            radii_norm = np.linalg.norm(pos, axis=1, keepdims=True)
            radii_norm[radii_norm == 0] = 1.0
            unit_inward = -pos / radii_norm
            forces = np.where(boundary_mask[:, np.newaxis], unit_inward, 0.0)
            
            pos += forces * step_size_ls

        # Project outside
        radii = np.linalg.norm(pos, axis=1)
        outside_mask = radii > 1.0
        if np.any(outside_mask):
            pos[outside_mask] = pos[outside_mask] / radii[outside_mask, np.newaxis]

        r_final = get_radius(pos)
        if r_final > best_r:
            best_r = r_final
            best_pos = pos.copy()

    return best_pos
