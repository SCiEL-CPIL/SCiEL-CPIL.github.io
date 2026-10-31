"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Initial random configuration
    def random_config():
        centers = np.random.rand(N, 2) * 0.8 + 0.1
        radii = np.random.rand(N) * 0.05 + 0.02
        return centers, radii
    
    def compute_score(centers, radii):
        # Check validity: circles inside [0,1]^2 and non-overlapping
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return 0.0
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    return 0.0
        return np.sum(radii)
    
    def is_valid(centers, radii):
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True
    
    def local_search(centers, radii):
        """Simple local search: try to increase radii and adjust positions"""
        best_centers = centers.copy()
        best_radii = radii.copy()
        best_score = compute_score(best_centers, best_radii)
        
        if best_score == 0:
            # Try to fix invalid configuration
            for _ in range(1000):
                if time.time() > deadline:
                    return best_centers, best_radii, 0.0
                # Random perturbation
                centers = best_centers + np.random.randn(N, 2) * 0.01
                radii = np.clip(best_radii + np.random.randn(N) * 0.005, 0.01, 0.3)
                # Project to valid region
                for i in range(N):
                    radii[i] = min(radii[i], min(centers[i][0], 1-centers[i][0], centers[i][1], 1-centers[i][1]))
                if is_valid(centers, radii):
                    score = compute_score(centers, radii)
                    if score > best_score:
                        best_centers = centers.copy()
                        best_radii = radii.copy()
                        best_score = score
            return best_centers, best_radii, best_score
        
        # Iteratively try to increase radii
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Try increasing each radius slightly
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try to increase radius of circle i
                new_radii = best_radii.copy()
                new_radii[i] += 0.002
                new_radii[i] = min(new_radii[i], min(best_centers[i][0], 1-best_centers[i][0], best_centers[i][1], 1-best_centers[i][1]))
                
                if new_radii[i] <= best_radii[i]:
                    continue
                
                # Check if this causes overlap
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.linalg.norm(best_centers[i] - best_centers[j])
                    if dist < new_radii[i] + best_radii[j] - 1e-9:
                        valid = False
                        break
                
                if valid and is_valid(best_centers, new_radii):
                    best_radii = new_radii
                    best_score = compute_score(best_centers, best_radii)
                    improved = True
            
            # Try moving circles to free up space
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try small random moves
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    new_centers = best_centers.copy()
                    new_centers[i] += np.random.randn(2) * 0.005
                    new_centers[i] = np.clip(new_centers[i], best_radii[i], 1-best_radii[i])
                    
                    if is_valid(new_centers, best_radii):
                        # See if we can now increase some radii
                        new_radii = best_radii.copy()
                        for j in range(N):
                            max_r = min(new_centers[j][0], 1-new_centers[j][0], new_centers[j][1], 1-new_centers[j][1])
                            # Check against other circles
                            for k in range(N):
                                if j == k:
                                    continue
                                dist = np.linalg.norm(new_centers[j] - new_centers[k])
                                max_r = min(max_r, dist - new_radii[k])
                            if max_r > new_radii[j] + 1e-6:
                                new_radii[j] = max_r
                            else:
                                new_radii[j] = min(new_radii[j], max_r)
                        
                        if is_valid(new_centers, new_radii):
                            score = compute_score(new_centers, new_radii)
                            if score > best_score:
                                best_centers = new_centers
                                best_radii = new_radii
                                best_score = score
                                improved = True
        
        return best_centers, best_radii, best_score
    
    # Multi-start local search
    best_score = 0.0
    best_centers = None
    best_radii = None
    
    # Start with a few different initial configurations
    num_starts = 20
    for start in range(num_starts):
        if time.time() > deadline:
            break
        
        if start == 0:
            # Grid-like initial
            side = int(np.ceil(np.sqrt(N)))
            centers = np.array([((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)])[:N]
            radii = np.full(N, 0.5 / side - 0.01)
        else:
            centers, radii = random_config()
        
        centers, radii, score = local_search(centers, radii)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    # Final optimization: try to squeeze more
    if best_centers is not None and best_score > 0:
        # Try greedy radius increase with position adjustment
        for _ in range(2000):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                
                # Try to increase radius of circle i
                old_r = best_radii[i]
                # Max possible radius based on boundaries
                max_boundary = min(best_centers[i][0], 1-best_centers[i][0], best_centers[i][1], 1-best_centers[i][1])
                
                # Max possible radius based on other circles
                max_other = max_boundary
                for j in range(N):
                    if i == j:
                        continue
                    dist = np.linalg.norm(best_centers[i] - best_centers[j])
                    max_other = min(max_other, dist - best_radii[j])
                
                if max_other <= old_r:
                    continue
                
                # Try a fraction of the increase
                for frac in [0.5, 0.75, 1.0]:
                    new_r = old_r + (max_other - old_r) * frac
                    new_radii = best_radii.copy()
                    new_radii[i] = new_r
                    
                    if is_valid(best_centers, new_radii):
                        best_radii = new_radii
                        best_score = compute_score(best_centers, best_radii)
                        improved = True
                        break
            
            if not improved:
                # Try moving a random circle
                i = random.randint(0, N-1)
                new_centers = best_centers.copy()
                new_centers[i] += np.random.randn(2) * 0.01
                new_centers[i] = np.clip(new_centers[i], best_radii[i], 1-best_radii[i])
                
                if is_valid(new_centers, best_radii):
                    best_centers = new_centers
                    # Try to increase radii again
                    for j in range(N):
                        max_boundary = min(best_centers[j][0], 1-best_centers[j][0], best_centers[j][1], 1-best_centers[j][1])
                        max_other = max_boundary
                        for k in range(N):
                            if j == k:
                                continue
                            dist = np.linalg.norm(best_centers[j] - best_centers[k])
                            max_other = min(max_other, dist - best_radii[k])
                        if max_other > best_radii[j]:
                            best_radii[j] = max_other
                    best_score = compute_score(best_centers, best_radii)
    
    # Construct output
    if best_centers is None or best_score == 0:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    result = np.column_stack([best_centers, best_radii])
    return result
# EVOLVE-BLOCK-END
