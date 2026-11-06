import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    # Seed random for reproducibility of the search path
    random.seed(42)
    
    # Initial configuration: 6x5 grid
    # Centered to allow moves in all directions
    # x: -2 to 3 (6 values)
    # y: -2 to 2 (5 values)
    points = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    assert len(points) == 30
    assert len(set(points)) == 30
    
    pts = np.array(points, dtype=int)
    
    # Precompute triple indices for vectorized scoring
    # There are C(30, 3) = 4060 triples
    indices = np.array(list(itertools.combinations(range(30), 3)))
    
    def get_score(p_array):
        # p_array is (30, 2)
        p_i = p_array[indices[:, 0]]
        p_j = p_array[indices[:, 1]]
        p_k = p_array[indices[:, 2]]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        
        # Determinant
        areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        # Exclude zero area
        non_zero_areas = areas[areas > 0]
        if len(non_zero_areas) == 0:
            return 0
            
        # Count multiplicities
        counts = np.bincount(non_zero_areas)
        return counts.max()

    current_score = get_score(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    # Define neighborhood moves
    # Small moves are more likely to improve local structure
    moves = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)]
    
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Hill Climbing Pass
        for i in range(30):
            if time.time() - start_time > time_limit:
                break
                
            x, y = pts[i]
            
            # Build occupied set for validity check
            # O(N) cost, 30 items, negligible
            occupied = set(map(tuple, pts))
            occupied.remove((x, y))
            
            # Shuffle moves to avoid bias
            random.shuffle(moves)
            
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                if (nx, ny) in occupied:
                    continue
                
                # Apply move
                pts[i] = (nx, ny)
                new_score = get_score(pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = pts.copy()
                    improved = True
                    # Don't break, keep looking for more improvements from this new state
                else:
                    # Revert
                    pts[i] = (x, y)
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck for a few passes, try perturbation
            if no_improve_count > 2:
                # Perturbation: move 2 random points to random valid locations
                i1, i2 = random.sample(range(30), 2)
                orig1 = pts[i1].copy()
                orig2 = pts[i2].copy()
                
                perturbed = False
                for _ in range(20): # 20 random attempts
                    nx1, ny1 = random.randint(-30, 30), random.randint(-30, 30)
                    nx2, ny2 = random.randint(-30, 30), random.randint(-30, 30)
                    
                    pts[i1] = (nx1, ny1)
                    pts[i2] = (nx2, ny2)
                    
                    # Check distinctness
                    if len(set(map(tuple, pts))) < 30:
                        continue
                    if (nx1, ny1) == (nx2, ny2):
                        continue
                    
                    new_score = get_score(pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = pts.copy()
                        perturbed = True
                        break
                    else:
                        # Revert
                        pts[i1] = orig1
                        pts[i2] = orig2
                
                if not perturbed:
                    # Revert if no improvement found in perturbation
                    pts[i1] = orig1
                    pts[i2] = orig2
                    # Reset counter to allow more perturbation attempts later?
                    # Or break if we've tried too many?
                    # Let's just continue, but maybe increase no_improve_count
                    no_improve_count = 0 # Reset to try again next pass
                else:
                    no_improve_count = 0

    return [tuple(p) for p in best_pts]
