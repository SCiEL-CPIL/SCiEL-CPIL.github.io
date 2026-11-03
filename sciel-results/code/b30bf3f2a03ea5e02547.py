import numpy as np
import time
from scipy.spatial.distance import cdist

N = 26

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None
    
    def get_energy(z):
        """Compute energy (penalty) for configuration z of shape (N, 3).
        Returns (energy, sum_r) where energy is 0 for valid configurations."""
        x, y, r = z[:, 0], z[:, 1], z[:, 2]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Pairwise distances
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i = r[iu[0]]
        r_j = r[iu[1]]
        
        # Overlap penalty (quadratic)
        overlaps = np.maximum(0, r_i + r_j - d_ij)
        penalty_overlap = np.sum(overlaps**2)
        
        # Boundary penalty (quadratic)
        pen_b = 0.0
        pen_b += np.sum(np.maximum(0, r - x)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - x))**2)
        pen_b += np.sum(np.maximum(0, r - y)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - y))**2)
        
        total_penalty = penalty_overlap + pen_b
        return total_penalty, sum_r
    
    def validate(z_opt):
        """Check if solution is valid and return sum of radii if so."""
        x, y, r = z_opt[:, 0], z_opt[:, 1], z_opt[:, 2]
        
        # Check radii are positive
        if np.any(r <= 0):
            return -np.inf
        
        # Pairwise distances
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i, r_j = r[iu[0]], r[iu[1]]
        max_overlap = np.max(np.maximum(0, r_i + r_j - d_ij))
        
        # Boundary check
        pen_b = 0.0
        pen_b += np.max(np.maximum(0, r - x))
        pen_b += np.max(np.maximum(0, r - (1 - x)))
        pen_b += np.max(np.maximum(0, r - y))
        pen_b += np.max(np.maximum(0, r - (1 - y)))
        
        if max_overlap < 1e-6 and pen_b < 1e-6:
            return np.sum(r)
        return -np.inf
    
    def simulated_annealing(z0, max_iters=5000, T_start=0.1, T_end=0.001, alpha=0.9995):
        """Run simulated annealing from initial configuration z0."""
        z = z0.copy()
        current_energy, _ = get_energy(z)
        best_z = z.copy()
        best_energy = current_energy
        best_sum_r = 0.0
        
        # Precompute upper triangle indices for speed
        iu = np.triu_indices(N, k=1)
        
        for it in range(max_iters):
            # Check time
            if time.time() - start_time > timeout:
                break
            
            # Temperature schedule
            T = T_start * (T_end / T_start) ** (it / max_iters)
            
            # Generate perturbation: move one circle
            idx = np.random.randint(N)
            
            # Perturbation magnitudes
            dx = np.random.randn() * 0.02
            dy = np.random.randn() * 0.02
            dr = np.random.randn() * 0.01
            
            # Propose new position
            z_new = z.copy()
            z_new[idx, 0] = np.clip(z[idx, 0] + dx, 0.0, 1.0)
            z_new[idx, 1] = np.clip(z[idx, 1] + dy, 0.0, 1.0)
            z_new[idx, 2] = np.clip(z[idx, 2] + dr, 0.01, 0.5)
            
            # Compute new energy
            new_energy, new_sum_r = get_energy(z_new)
            
            # Accept if better or with probability exp(-delta/T)
            delta = new_energy - current_energy
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                z = z_new
                current_energy = new_energy
                
                # Track best valid solution
                if current_energy < 1e-8:
                    if new_sum_r > best_sum_r:
                        best_sum_r = new_sum_r
                        best_z = z.copy()
                        best_energy = current_energy
            
            # Occasionally do a larger perturbation
            if it % 100 == 0:
                for _ in range(3):
                    idx = np.random.randint(N)
                    dx = np.random.randn() * 0.05
                    dy = np.random.randn() * 0.05
                    dr = np.random.randn() * 0.02
                    z[idx, 0] = np.clip(z[idx, 0] + dx, 0.0, 1.0)
                    z[idx, 1] = np.clip(z[idx, 1] + dy, 0.0, 1.0)
                    z[idx, 2] = np.clip(z[idx, 2] + dr, 0.01, 0.5)
        
        return best_z, best_sum_r
    
    def local_refine(z):
        """Local refinement using small perturbations to fine-tune."""
        z = z.copy()
        current_energy, current_sum_r = get_energy(z)
        
        for _ in range(200):
            if time.time() - start_time > timeout:
                break
            
            improved = False
            for idx in range(N):
                # Try small moves in each direction
                for dx, dy, dr in [(0.005, 0, 0), (-0.005, 0, 0), (0, 0.005, 0), (0, -0.005, 0), 
                                    (0, 0, 0.003), (0, 0, -0.003)]:
                    z_new = z.copy()
                    z_new[idx, 0] = np.clip(z[idx, 0] + dx, 0.0, 1.0)
                    z_new[idx, 1] = np.clip(z[idx, 1] + dy, 0.0, 1.0)
                    z_new[idx, 2] = np.clip(z[idx, 2] + dr, 0.01, 0.5)
                    
                    new_energy, new_sum_r = get_energy(z_new)
                    
                    # Accept if valid and improves sum_r, or if it reduces energy
                    if new_energy < 1e-8 and new_sum_r > current_sum_r:
                        z = z_new
                        current_energy = new_energy
                        current_sum_r = new_sum_r
                        improved = True
                    elif new_energy < current_energy:
                        z = z_new
                        current_energy = new_energy
                        current_sum_r = new_sum_r
                        improved = True
            
            if not improved:
                break
        
        return z, current_sum_r
    
    # Generate diverse initial configurations
    inits = []
    
    # Init 1: Hexagonal packing
    def generate_hex_init(scale, offset_x, offset_y):
        pts = []
        row = 0
        while True:
            y = row * scale * np.sqrt(3) / 2 + offset_y
            if y > 1.0:
                break
            col = 0
            while True:
                x = col * scale + offset_x + (row % 2) * scale / 2
                if x > 1.0:
                    break
                pts.append((x, y))
                col += 1
            row += 1
        
        if len(pts) >= N:
            np.random.shuffle(pts)
            selected = pts[:N]
            init = np.zeros((N, 3))
            for i, (x, y) in enumerate(selected):
                init[i, 0] = x
                init[i, 1] = y
                init[i, 2] = 0.08
            return init
        return None
    
    # Multiple hex configurations
    for scale in [0.14, 0.16, 0.18]:
        for ox in [0.02, 0.08]:
            for oy in [0.02, 0.08]:
                init = generate_hex_init(scale, ox, oy)
                if init is not None:
                    inits.append(init)
    
    # Init 2: Random configurations
    for seed in range(15):
        np.random.seed(seed)
        rx = np.random.rand(N)
        ry = np.random.rand(N)
        rr = np.full(N, 0.07)
        inits.append(np.column_stack((rx, ry, rr)))
    
    # Init 3: Grid-based
    side = 5
    for offset in [0.1, 0.15]:
        init = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    init[count, 0] = (i + 0.5) / side + offset * np.random.randn()
                    init[count, 1] = (j + 0.5) / side + offset * np.random.randn()
                    init[count, 2] = 0.07
                    count += 1
        if count == N:
            inits.append(init)
    
    # Run SA + local refinement from each init
    for init in inits:
        if time.time() - start_time > timeout:
            break
        
        try:
            # Simulated annealing
            z_sa, sum_r_sa = simulated_annealing(init, max_iters=3000)
            
            # Local refinement
            z_final, sum_r_final = local_refine(z_sa)
            
            # Validate and update best
            score = validate(z_final)
            if score > best_score:
                best_score = score
                best_solution = z_final
        except Exception:
            pass
    
    # If no valid solution found, try a simple fallback
    if best_solution is None:
        # Fallback: place circles in a grid with small radii
        side = int(np.ceil(np.sqrt(N)))
        best_solution = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    best_solution[count, 0] = (i + 0.5) / side
                    best_solution[count, 1] = (j + 0.5) / side
                    best_solution[count, 2] = 0.5 / side - 0.01
                    count += 1
    
    return best_solution
