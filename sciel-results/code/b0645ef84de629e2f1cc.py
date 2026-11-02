"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np

N = 200

def construct():
    def hexagonal_init(n_circles):
        """Generate hexagonal lattice points centered in unit square."""
        best_points = None
        best_r = 0
        
        # Try different numbers of rows
        for rows in range(10, 25):
            for cols in range(10, 25):
                total = rows * cols
                if total < n_circles:
                    continue
                
                # Hexagonal grid dimensions
                # Horizontal spacing: 2r
                # Vertical spacing: r * sqrt(3)
                # With offset, the effective width is (cols-1)*2r + r = (2*cols-1)*r
                # Height: (rows-1)*r*sqrt(3)
                
                # Fit in square [-0.5, 0.5]^2
                # (2*cols-1)*r <= 1 => r <= 1/(2*cols-1)
                # (rows-1)*r*sqrt(3) <= 1 => r <= 1/((rows-1)*sqrt(3))
                
                r_max = min(1.0 / (2*cols - 1), 1.0 / ((rows - 1) * np.sqrt(3)))
                
                if r_max > best_r:
                    best_r = r_max
                    # Generate points
                    pts = []
                    for j in range(rows):
                        for i in range(cols):
                            x = (i * 2 - (cols - 1)) * r_max  # centered
                            if j % 2 == 1:
                                x += r_max  # offset
                            y = (j - (rows - 1) / 2.0) * r_max * np.sqrt(3)
                            pts.append([x, y])
                    pts = np.array(pts[:n_circles])
                    # Check if all points are within bounds
                    if np.all(np.abs(pts) <= 0.5 - r_max * 0.01):
                        best_points = pts
                        best_r = r_max
        
        if best_points is not None:
            return best_points
        
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]

    def simulate(centers_init, steps=3000, dt=0.002, damping=0.98, k_rep=50.0, k_bound=50.0):
        centers = centers_init.copy()
        velocities = np.zeros_like(centers)
        
        for step in range(steps):
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            np.fill_diagonal(dists, np.inf)
            
            # F = k * diff / dist^3
            inv_d3 = 1.0 / (dists_sq * dists)
            force_contrib = diff * inv_d3[:, :, np.newaxis]
            forces += k_rep * np.sum(force_contrib, axis=1)
            
            # Boundary forces
            d_left = np.maximum(centers[:, 0] + 0.5, 1e-6)
            d_right = np.maximum(0.5 - centers[:, 0], 1e-6)
            d_bottom = np.maximum(centers[:, 1] + 0.5, 1e-6)
            d_top = np.maximum(0.5 - centers[:, 1], 1e-6)
            
            F_x_bound = k_bound * (1.0 / d_left**2 - 1.0 / d_right**2)
            F_y_bound = k_bound * (1.0 / d_bottom**2 - 1.0 / d_top**2)
            
            forces[:, 0] += F_x_bound
            forces[:, 1] += F_y_bound
            
            # Symplectic Euler
            velocities += forces * dt
            velocities *= damping
            centers += velocities * dt
            
            # Clamp
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
        return centers

    def compute_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2
        
        bound_dists = np.minimum(
            np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
            np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
        )
        min_bound = np.min(bound_dists)
        
        return min(min_pair, min_bound)

    # Run multiple simulations with different initializations
    best_centers = None
    best_radius = 0.0
    
    # 1. Hexagonal init
    centers_hex = hexagonal_init(N)
    sim_centers = simulate(centers_hex)
    r_sim = compute_radius(sim_centers)
    
    if r_sim > best_radius:
        best_radius = r_sim
        best_centers = sim_centers
    
    # 2. Square init
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers_sq = grid[:N]
    sim_centers_sq = simulate(centers_sq)
    r_sim_sq = compute_radius(sim_centers_sq)
    
    if r_sim_sq > best_radius:
        best_radius = r_sim_sq
        best_centers = sim_centers_sq
        
    # 3. Random perturbations of the best so far
    for _ in range(5):
        noise = np.random.normal(0, 0.01, best_centers.shape)
        perturbed = best_centers + noise
        # Clip to bounds
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.5, 0.5)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.5, 0.5)
        
        sim_centers_pert = simulate(perturbed, steps=2000)
        r_sim_pert = compute_radius(sim_centers_pert)
        
        if r_sim_pert > best_radius:
            best_radius = r_sim_pert
            best_centers = sim_centers_pert

    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    # Compute radius for display
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    bound_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_bound = np.min(bound_dists)
    print(f"Radius: {min(min_pair, min_bound):.10f}")
