import numpy as np
import random
import math
from scipy.spatial import cKDTree

N = 12
GRID_SPACING = 0.1
MIN_DIST = math.sqrt(3) - 1e-6

def construct():
    # Generate hex grid
    h = GRID_SPACING
    grid_points = []
    for r in range(-60, 61):
        y = h * r * math.sqrt(3) / 2
        if abs(y) > 5: continue
        for q in range(-60, 61):
            x = h * (q + r / 2.0)
            if abs(x) > 5: continue
            grid_points.append((x, y))
            
    grid_points = np.array(grid_points)
    num_points = len(grid_points)
    
    # Build KDTree for neighbor lookup
    tree = cKDTree(grid_points)
    
    # Pre-compute neighbors (within radius 0.5)
    # We will use this to pick moves
    # Actually, we can just pick a random neighbor on the fly using tree.query_ball_point
    # But that's O(log N + k).
    
    # Normals
    angles = [math.pi/6 + i * math.pi/3 for i in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in angles])
    
    def get_cost(indices):
        # indices: array of N ints
        points = grid_points[indices]
        
        # 1. Check overlaps
        penalty = 0.0
        for i in range(N):
            pi = points[i]
            for j in range(i+1, N):
                pj = points[j]
                d = np.linalg.norm(pi - pj)
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) * 10.0
                    
        if penalty > 0:
            return penalty + 100.0
            
        # 2. Calculate S
        projs = points @ normals.T
        maxes = np.max(projs, axis=0)
        mins = np.min(projs, axis=0)
        widths = maxes - mins + 2.0
        max_width = np.max(widths)
        S = max_width / math.sqrt(3)
        
        return S

    best_S = float('inf')
    best_indices = None
    
    for start in range(30):
        # Init
        current_indices = np.array(random.sample(range(num_points), N))
        current_cost = get_cost(current_indices)
        
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        max_iters = 1000
        
        for _ in range(max_iters):
            if T < min_T: break
            
            idx = random.randint(0, N-1)
            old_idx = current_indices[idx]
            
            # Get neighbors of old_idx
            # Use a radius that allows movement
            # Grid spacing 0.1. Neighbors are ~0.1 away.
            # Let's pick a random neighbor within 0.5
            neighbors = tree.query_ball_point(grid_points[old_idx], r=0.5)
            
            if len(neighbors) > 1:
                new_idx = random.choice(neighbors)
                # Ensure new_idx is not already in current_indices (other than old_idx)
                # If it is, we are swapping? No, we are moving one point.
                # If new_idx is in current_indices, we would have duplicate points.
                # We must ensure uniqueness.
                if new_idx != old_idx and new_idx not in current_indices:
                    current_indices[idx] = new_idx
                    new_cost = get_cost(current_indices)
                    
                    if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost)/T):
                        current_cost = new_cost
                        if new_cost < best_S:
                            best_S = new_cost
                            best_indices = current_indices.copy()
                    else:
                        current_indices[idx] = old_idx
            else:
                # No neighbors? Pick random point
                new_idx = random.randint(0, num_points-1)
                if new_idx != old_idx and new_idx not in current_indices:
                    current_indices[idx] = new_idx
                    new_cost = get_cost(current_indices)
                    if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost)/T):
                        current_cost = new_cost
                        if new_cost < best_S:
                            best_S = new_cost
                            best_indices = current_indices.copy()
                    else:
                        current_indices[idx] = old_idx
                        
            T *= cooling
            
    # Return best
    if best_indices is None:
        # Fallback
        best_indices = np.array(random.sample(range(num_points), N))
        
    inner = []
    for idx in best_indices:
        p = grid_points[idx]
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    return inner, (0.0, 0.0), float(best_S), 0.0
