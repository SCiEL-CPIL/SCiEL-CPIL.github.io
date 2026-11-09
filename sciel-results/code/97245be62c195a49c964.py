"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from itertools import combinations

def construct():
    """
    Construct a valid 2-coloring of triples on n vertices with no all-red K4^(3)
    and no all-blue K5^(3). Uses optimized local search.
    """
    target_n = 34
    time_limit = 110  # seconds
    
    # Precompute all triples
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    
    # For each 4-set, store the 4 triple indices
    k4_table = []
    for quad in combinations(range(target_n), 4):
        quad_set = set(quad)
        tri_indices = []
        for i, (a, b, c) in enumerate(triples):
            if a in quad_set and b in quad_set and c in quad_set:
                tri_indices.append(i)
        if len(tri_indices) == 4:
            k4_table.append(tri_indices)
    
    # For each 5-set, store the 10 triple indices
    k5_table = []
    for quint in combinations(range(target_n), 5):
        quint_set = set(quint)
        tri_indices = []
        for i, (a, b, c) in enumerate(triples):
            if a in quint_set and b in quint_set and c in quint_set:
                tri_indices.append(i)
        if len(tri_indices) == 10:
            k5_table.append(tri_indices)
    
    num_k4s = len(k4_table)
    num_k5s = len(k5_table)
    
    # Precompute: for each K4, which triples are red (we track this)
    # Use a faster violation counting method
    
    def count_violations(col):
        """Count red K4 violations and blue K5 violations."""
        red_v = 0
        for tri_indices in k4_table:
            # Check if all 4 are red (0)
            if col[tri_indices[0]] == 0 and col[tri_indices[1]] == 0 and col[tri_indices[2]] == 0 and col[tri_indices[3]] == 0:
                red_v += 1
        blue_v = 0
        for tri_indices in k5_table:
            # Check if all 10 are blue (1)
            if (col[tri_indices[0]] == 1 and col[tri_indices[1]] == 1 and col[tri_indices[2]] == 1 and 
                col[tri_indices[3]] == 1 and col[tri_indices[4]] == 1 and col[tri_indices[5]] == 1 and 
                col[tri_indices[6]] == 1 and col[tri_indices[7]] == 1 and col[tri_indices[8]] == 1 and 
                col[tri_indices[9]] == 1):
                blue_v += 1
        return red_v, blue_v
    
    # Try to find a valid coloring
    start_time = time.time()
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random coloring
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.uint8)
        
        for iteration in range(20000):
            if time.time() - start_time > time_limit:
                break
            
            red_v, blue_v = count_violations(colors)
            
            if red_v == 0 and blue_v == 0:
                return {'n': target_n, 'colors': colors.tolist()}
            
            # Pick a violation to fix
            if red_v > 0 and (blue_v == 0 or random.random() < 0.5):
                # Find a violated K4 and flip one of its triples
                for tri_indices in k4_table:
                    if (colors[tri_indices[0]] == 0 and colors[tri_indices[1]] == 0 and 
                        colors[tri_indices[2]] == 0 and colors[tri_indices[3]] == 0):
                        idx = random.choice(tri_indices)
                        colors[idx] ^= 1
                        break
            else:
                # Find a violated K5 and flip one of its triples
                for tri_indices in k5_table:
                    if (colors[tri_indices[0]] == 1 and colors[tri_indices[1]] == 1 and colors[tri_indices[2]] == 1 and 
                        colors[tri_indices[3]] == 1 and colors[tri_indices[4]] == 1 and colors[tri_indices[5]] == 1 and 
                        colors[tri_indices[6]] == 1 and colors[tri_indices[7]] == 1 and colors[tri_indices[8]] == 1 and 
                        colors[tri_indices[9]] == 1):
                        idx = random.choice(tri_indices)
                        colors[idx] ^= 1
                        break
    
    # Fallback: use sum mod 3 for small n
    for test_n in range(34, 0, -1):
        if time.time() - start_time > time_limit:
            break
        tris = list(combinations(range(test_n), 3))
        cols = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in tris]
        
        # Quick validity check
        valid = True
        # Check K4
        for quad in combinations(range(test_n), 4):
            quad_set = set(quad)
            all_red = True
            for tri in combinations(quad, 3):
                # Find this triple's index
                for j, t in enumerate(tris):
                    if t == tri:
                        if cols[j] != 0:
                            all_red = False
                            break
                        break
                if not all_red:
                    break
            if all_red:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check K5
        for quint in combinations(range(test_n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                for j, t in enumerate(tris):
                    if t == tri:
                        if cols[j] != 1:
                            all_blue = False
                            break
                        break
                if not all_blue:
                    break
            if all_blue:
                valid = False
                break
        
        if valid:
            return {'n': test_n, 'colors': cols}
    
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
