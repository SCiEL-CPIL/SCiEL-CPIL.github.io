import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    # ... existing candidates ...
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)
    
    # ... (keep all existing candidates) ...
    # I will add a generator for more candidates

    def generate_compact_candidates(num_to_generate=50):
        gen_candidates = []
        # Directions in axial coordinates
        dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        
        for _ in range(num_to_generate):
            # Start with center
            current = [(0, 0)]
            current_set = set(current)
            
            # Boundary: cells not in current_set but adjacent to current_set
            # We will greedily add cells that keep the shape compact
            
            for _ in range(N - 1):
                # Find all valid neighbors
                valid_neighbors = []
                for q, r in current:
                    for dq, dr in dirs:
                        nq, nr = q + dq, r + dr
                        if (nq, nr) not in current_set:
                            valid_neighbors.append((nq, nr))
                
                if not valid_neighbors:
                    break
                
                # Score each neighbor: minimize distance to centroid
                cx = sum(p[0] for p in current) / len(current)
                cy = sum(p[1] for p in current) / len(current)
                
                best_neighbor = None
                min_dist = float('inf')
                
                # To introduce diversity, we don't always pick the absolute best.
                # But for "compact", let's pick the best with some randomness.
                # Or just pick the best to get very compact shapes.
                
                for nq, nr in valid_neighbors:
                    # Axial to cartesian distance approx
                    x = math.sqrt(3) * (nq + nr / 2.0)
                    y = 1.5 * nr
                    dist = (x - math.sqrt(3) * (cx + cy / 2.0))**2 + (y - 1.5 * cy)**2
                    
                    # Add small random noise to break ties and allow diversity
                    dist += np.random.uniform(0, 0.1)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_neighbor = (nq, nr)
                        
                if best_neighbor:
                    current.append(best_neighbor)
                    current_set.add(best_neighbor)
                    
            if len(current) == N:
                # Check if this candidate is unique (sorted tuple of tuples)
                # Simple check: compare with existing
                is_unique = True
                key = tuple(sorted(current))
                for existing in gen_candidates:
                    if tuple(sorted(existing)) == key:
                        is_unique = False
                        break
                if is_unique:
                    gen_candidates.append(current)
                    
        return gen_candidates

    # Generate new candidates
    try:
        new_candidates = generate_compact_candidates(100)
        candidates.extend(new_candidates)
    except Exception:
        pass

    # ... rest of the code ...
