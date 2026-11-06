"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from collections import Counter

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    # The seed returns [(x,y) for x in range(6) for y in range(5)]
    # which is a 6x5 grid. Let's compute its score to establish baseline.
    # Actually, let's just start with a good heuristic and improve.
    
    # Strategy:
    # 1. Start with a structured grid (like the seed) but maybe larger or different.
    # 2. Use local search / simulated annealing to improve the configuration.
    # 3. Objective: maximize the maximum multiplicity of any single positive doubled area.
    
    # Let's define a fast scoring function.
    def compute_scores(points):
        """Compute all doubled areas (absolute determinant) for all triples."""
        n = len(points)
        if n < 3:
            return {}
        areas = []
        # Using numpy for speed
        pts = np.array(points)
        for i in range(n):
            for j in range(i+1, n):
                # Vector from i to j
                v1 = pts[j] - pts[i]
                # For each k > j, compute cross product with vector from i to k
                for k in range(j+1, n):
                    v2 = pts[k] - pts[i]
                    det = abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if det > 0:
                        areas.append(det)
        if not areas:
            return {}
        counter = Counter(areas)
        return counter
    
    def get_max_multiplicity(points):
        """Return the maximum multiplicity of any single positive doubled area."""
        counter = compute_scores(points)
        if not counter:
            return 0
        return max(counter.values())
    
    # Initial configuration: try a few structured grids
    # The seed is 6x5 = 30 points. Let's try that first.
    best_points = [(x, y) for x in range(6) for y in range(5)]
    best_score = get_max_multiplicity(best_points)
    
    # Try other structured configurations
    # A 5x6 grid (transposed)
    alt1 = [(x, y) for x in range(5) for y in range(6)]
    s1 = get_max_multiplicity(alt1)
    if s1 > best_score:
        best_score = s1
        best_points = alt1
    
    # Try a 3x10 grid
    alt2 = [(x, y) for x in range(3) for y in range(10)]
    s2 = get_max_multiplicity(alt2)
    if s2 > best_score:
        best_score = s2
        best_points = alt2
    
    # Try a 10x3 grid
    alt3 = [(x, y) for x in range(10) for y in range(3)]
    s3 = get_max_multiplicity(alt3)
    if s3 > best_score:
        best_score = s3
        best_points = alt3
    
    # Try a 4x8 grid (32 points, need to remove 2)
    # Actually, let's stick with exactly 30 points.
    
    # Now do local search / hill climbing
    # We'll try to swap points or move points to improve the score.
    
    # For efficiency, let's precompute areas more cleverly.
    # But first, let's see how long the scoring takes.
    
    # Local search: try moving each point to a nearby location
    # and see if the score improves.
    
    # We need to be careful about the time budget.
    
    # Let's implement a faster scoring that can be updated incrementally.
    # But for simplicity, let's first try a simple hill climbing.
    
    # Actually, let's think about what configurations maximize equal-area triangles.
    # A regular lattice tends to produce many equal-area triangles.
    # The 6x5 grid is a good starting point.
    
    # Let's try to optimize the grid dimensions.
    # For a WxH grid with W*H = 30, the possible factor pairs are:
    # (1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)
    
    # We already tried (6,5), (5,6), (3,10), (10,3).
    # Let's try (2,15) and (15,2).
    
    alt4 = [(x, y) for x in range(2) for y in range(15)]
    s4 = get_max_multiplicity(alt4)
    if s4 > best_score:
        best_score = s4
        best_points = alt4
    
    alt5 = [(x, y) for x in range(15) for y in range(2)]
    s5 = get_max_multiplicity(alt5)
    if s5 > best_score:
        best_score = s5
        best_points = alt5
    
    # Now let's do a local search to see if we can improve further.
    # We'll try perturbing the grid slightly.
    
    # Actually, let's try a different approach: start with a random configuration
    # and use simulated annealing or hill climbing.
    
    # But first, let's check the time.
    if time.time() > deadline - 10:
        return best_points
    
    # Let's try to improve the best grid configuration by local search.
    # We'll try moving each point to a nearby integer coordinate.
    
    # For efficiency, we'll use a neighborhood of radius 1 or 2.
    
    # Let's implement a simple hill climbing:
    # 1. For each point, try moving it to each of its 8 neighbors.
    # 2. If the new score is better, accept the move.
    # 3. Repeat until no improvement is found or time runs out.
    
    # But computing the full score each time is expensive.
    # Let's estimate: 30 points, C(30,3) = 4060 triples.
    # Each triple computation is O(1) with numpy.
    # So one scoring call is ~4060 operations, which is fast.
    
    # Let's do a few rounds of hill climbing.
    
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 5:
        improved = False
        iteration += 1
        
        for i in range(len(best_points)):
            if time.time() > deadline - 5:
                break
            x, y = best_points[i]
            
            # Try moving to nearby points
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    
                    # Check if the new point is already in the set
                    if (nx, ny) in best_points:
                        continue
                    
                    # Create new configuration
                    new_points = best_points.copy()
                    new_points[i] = (nx, ny)
                    
                    # Check for duplicates
                    if len(set(new_points)) != 30:
                        continue
                    
                    # Compute new score
                    new_score = get_max_multiplicity(new_points)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points
                        improved = True
    
    # Return the best configuration found
    return best_points
# EVOLVE-BLOCK-END
