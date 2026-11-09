"""Physics-based relaxation for packing 300 equal circles in a unit circle."""
import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    def generate_hex_lattice():
        """Generate initial hexagonal lattice points."""
        spacing = np.sqrt(4.0 * np.pi / N)
        k = int(3.0 / spacing) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) == 0:
            return np.random.randn(N, 2) * 0.5
        
        if len(points) > N:
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:N]
            points = points[idx]
        else:
            while len(points) < N:
                p = np.random.randn(2) * 0.5
                points = np.vstack([points, [p]])
        
        return points[:N]
    
    def score_config(pos):
        """Compute the packing score (minimum radius)."""
        # Compute all pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        radii = np.linalg.norm(pos, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        
        if min_boundary_dist < 0:
            return 0.0
        
        return min(min_pair_dist, min_boundary_dist) / 2.0
    
    # Initialize
    pos = generate_hex_lattice()
    best_pos = pos.copy()
    best_score = score_config(pos)
    
    # Gradient-based optimization with simulated annealing
    temp = 0.02
    cooling = 0.9995
    lr = 0.005
    
    def compute_gradients(pos):
        """Compute gradient of negative score (to maximize score)."""
        n = len(pos)
        grads = np.zeros_like(pos)
        
        # Pairwise repulsion: push apart if too close
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find close pairs
        close_mask = dists < 0.15
        for i in range(n):
            for j in range(i + 1, n):
                if close_mask[i, j]:
                    d = pos[i] - pos[j]
                    dist = dists[i, j]
                    if dist > 1e-10:
                        f = 10.0 * (0.15 - dist) / dist
                        grads[i] += f * d
                        grads[j] -= f * d
        
        # Boundary constraint
        radii = np.linalg.norm(pos, axis=1)
        for i in range(n):
            if radii[i] > 0.95:
                grads[i] -= 5.0 * (radii[i] - 0.95) * pos[i] / radii[i]
        
        return grads
    
    iterations = 0
    while time.time() < deadline:
        # Compute gradient
        grads = compute_gradients(pos)
        
        # Update with noise
        noise = np.random.randn(N, 2) * 0.005 * np.sqrt(temp)
        pos = pos + lr * grads + noise
        
        # Hard constraint: keep inside unit circle
        radii = np.linalg.norm(pos, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            pos[mask] = pos[mask] / radii[mask, None] * 0.999
        
        # Compute score
        current_score = score_config(pos)
        
        if current_score > best_score:
            best_score = current_score
            best_pos = pos.copy()
        
        # Cool down
        temp = max(temp * cooling, 0.001)
        iterations += 1
    
    return best_pos
