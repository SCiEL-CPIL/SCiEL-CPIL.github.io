"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    np.random.seed(42)
    
    def check_validity(pos, rad):
        """Check if all circles are inside unit square and non-overlapping."""
        if np.any(pos[:, 0] - rad < -1e-9) or np.any(pos[:, 0] + rad > 1 + 1e-9):
            return False
        if np.any(pos[:, 1] - rad < -1e-9) or np.any(pos[:, 1] + rad > 1 + 1e-9):
            return False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                if dist < rad[i] + rad[j] - 1e-7:
                    return False
        return True
    
    def objective(params):
        """Negative sum of radii, with penalty for invalid configurations."""
        pos = params[:2*N].reshape(N, 2)
        rad = params[2*N:]
        
        penalty = 0.0
        
        # Boundary penalties
        for i in range(N):
            if pos[i,0] - rad[i] < 0:
                penalty += (pos[i,0] - rad[i])**2 * 1000
            if pos[i,0] + rad[i] > 1:
                penalty += (pos[i,0] + rad[i] - 1)**2 * 1000
            if pos[i,1] - rad[i] < 0:
                penalty += (pos[i,1] - rad[i])**2 * 1000
            if pos[i,1] + rad[i] > 1:
                penalty += (pos[i,1] + rad[i] - 1)**2 * 1000
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = rad[i] + rad[j] - dist
                if overlap > 0:
                    penalty += overlap**2 * 1000
        
        return -np.sum(rad) + penalty
    
    def local_search(pos, rad, max_iters=200):
        """Local search to refine the solution."""
        pos = pos.copy()
        rad = rad.copy()
        
        for _ in range(max_iters):
            improved = False
            
            # Try to increase radii
            for i in range(N):
                # Max possible radius for circle i
                max_r = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
                for j in range(N):
                    if i != j:
                        dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                        max_r = min(max_r, (dist - rad[j]))
                
                new_r = min(max_r, rad[i] * 1.01)
                if new_r > rad[i] + 1e-8:
                    # Check if increasing radius i maintains validity
                    valid = True
                    for j in range(N):
                        if i != j:
                            dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                            if dist < new_r + rad[j] - 1e-8:
                                valid = False
                                break
                    
                    if valid and new_r <= min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1]) + 1e-8:
                        rad[i] = new_r
                        improved = True
            
            # Try moving circles
            for i in range(N):
                for dx, dy in [(0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)]:
                    new_pos = pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    
                    if new_pos[i,0] < 0 or new_pos[i,0] > 1 or new_pos[i,1] < 0 or new_pos[i,1] > 1:
                        continue
                    
                    # Check if this move allows increasing some radii
                    valid = True
                    for j in range(N):
                        if i != j:
                            dist = np.sqrt((new_pos[i,0]-pos[j,0])**2 + (new_pos[i,1]-pos[j,1])**2)
                            if dist < rad[i] + rad[j] - 1e-8:
                                valid = False
                                break
                    
                    if valid:
                        pos = new_pos
                        improved = True
                        break
            
            if not improved:
                break
        
        return pos, rad
    
    # Generate multiple initial configurations
    best_score = np.inf
    best_params = None
    
    # Configuration 1: Grid-based
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    
    # Add remaining circles in gaps
    while len(positions) < N:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        positions.append([x, y])
    
    positions = np.array(positions[:N])
    radii = np.full(N, 0.08)
    
    # Configuration 2: Hexagonal-like
    positions2 = []
    for i in range(5):
        for j in range(5):
            if len(positions2) >= N:
                break
            x = (i + 0.5) / 5 + (0.1 if j % 2 == 0 else -0.1)
            y = (j + 0.5) / 5
            x = np.clip(x, 0.1, 0.9)
            positions2.append([x, y])
    
    while len(positions2) < N:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        positions2.append([x, y])
    
    positions2 = np.array(positions2[:N])
    radii2 = np.full(N, 0.08)
    
    # Configuration 3: Random
    positions3 = np.random.uniform(0.15, 0.85, size=(N, 2))
    radii3 = np.full(N, 0.07)
    
    configs = [(positions, radii), (positions2, radii2), (positions3, radii3)]
    
    for cfg_idx, (pos_init, rad_init) in enumerate(configs):
        for restart in range(3):
            if restart > 0:
                pos = pos_init + np.random.uniform(-0.1, 0.1, size=(N, 2))
                pos = np.clip(pos, 0.1, 0.9)
                rad = rad_init + np.random.uniform(-0.02, 0.02, size=N)
                rad = np.clip(rad, 0.03, 0.15)
            else:
                pos = pos_init.copy()
                rad = rad_init.copy()
            
            params = np.concatenate([pos.flatten(), rad])
            
            bounds = []
            for i in range(N):
                bounds.append((0, 1))  # x
                bounds.append((0, 1))  # y
            for i in range(N):
                bounds.append((0.001, 0.4))  # radius
            
            try:
                result = minimize(objective, params, method='L-BFGS-B', bounds=bounds, 
                                  options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
            except Exception:
                continue
    
    # Extract best solution
    if best_params is not None:
        best_pos = best_params[:2*N].reshape(N, 2)
        best_rad = best_params[2*N:]
        
        # Local search refinement
        best_pos, best_rad = local_search(best_pos, best_rad, max_iters=100)
        
        # Final validity check
        if not check_validity(best_pos, best_rad):
            # Try to fix by reducing radii slightly
            for i in range(N):
                best_rad[i] *= 0.95
            
            if not check_validity(best_pos, best_rad):
                # Fallback to simple grid
                side = 5
                cells = []
                for i in range(side):
                    for j in range(side):
                        if len(cells) >= N:
                            break
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        r = 0.5 / side - 0.01
                        cells.append([x, y, r])
                return np.array(cells)
        
        return np.column_stack([best_pos, best_rad])
    else:
        # Fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        return np.array(cells)
# EVOLVE-BLOCK-END

