"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute all K4s and their edge indices
    edge_idx = {}
    for idx, (a, b) in enumerate(edges):
        edge_idx[(a, b)] = idx
    
    k4s = []
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        e_indices = [
            edge_idx[(v0, v1)],
            edge_idx[(v0, v2)],
            edge_idx[(v0, v3)],
            edge_idx[(v1, v2)],
            edge_idx[(v1, v3)],
            edge_idx[(v2, v3)],
        ]
        k4s.append(e_indices)
    
    num_k4s = len(k4s)  # 1820
    
    def count_colors(col):
        return len(set(col))
    
    def count_violations(col):
        """Count number of K4s with < 5 distinct colors."""
        violations = 0
        for e_idx in k4s:
            colors_in_k4 = set(col[i] for i in e_idx)
            if len(colors_in_k4) < 5:
                violations += 1
        return violations
    
    def is_valid(col):
        return count_violations(col) == 0
    
    # Strategy: Start with (i+j) % 15, check validity, then try to reduce colors
    # If invalid, repair using local search targeting fewer colors
    
    best_result = None
    best_color_count = float('inf')
    
    # Try base colorings with different moduli
    for modulus in [15, 14, 13, 17, 16]:
        if time.time() > deadline - 15.0:
            break
        
        base_col = [(i + j) % modulus for i in range(n) for j in range(i + 1, n)]
        
        if is_valid(base_col):
            cc = count_colors(base_col)
            if cc < best_color_count:
                best_color_count = cc
                best_result = base_col[:]
        
        # If not valid or we can do better, try to improve via local search
        if time.time() > deadline - 10.0:
            break
        
        # Local search to fix violations and reduce colors
        col = base_col[:]
        current_violations = count_violations(col)
        current_colors = count_colors(col)
        
        if current_violations == 0 and current_colors < best_color_count:
            best_color_count = current_colors
            best_result = col[:]
            continue
        
        # Try to reduce to target colors
        target = min(current_colors, 15)
        
        # Local search with multiple restarts
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            
            col = base_col[:]
            current_violations = count_violations(col)
            current_colors = count_colors(col)
            
            if current_violations == 0:
                if current_colors < best_color_count:
                    best_color_count = current_colors
                    best_result = col[:]
                break
            
            # Simulated annealing style local search
            temperature = 1.0
            cooling = 0.999
            no_improve = 0
            max_iters = 3000
            
            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                
                # Pick a random edge
                e = random.randint(0, num_edges - 1)
                old_color = col[e]
                
                # Try a new color from the target palette
                new_color = random.randint(0, target - 1)
                if new_color == old_color:
                    continue
                
                col[e] = new_color
                new_violations = count_violations(col)
                new_colors = count_colors(col)
                
                # Acceptance criteria
                if new_violations < current_violations:
                    current_violations = new_violations
                    current_colors = new_colors
                    no_improve = 0
                elif new_violations == current_violations:
                    if new_colors < current_colors:
                        current_violations = new_violations
                        current_colors = new_colors
                        no_improve = 0
                    elif new_colors == current_colors and random.random() < temperature:
                        no_improve = 0
                    else:
                        col[e] = old_color
                else:
                    # Accept worse with some probability (annealing)
                    if random.random() < temperature * 0.1:
                        current_violations = new_violations
                        current_colors = new_colors
                    else:
                        col[e] = old_color
                
                temperature *= cooling
                
                if current_violations == 0:
                    break
                
                no_improve += 1
                if no_improve > 500:
                    break
            
            if current_violations == 0:
                if current_colors < best_color_count:
                    best_color_count = current_colors
                    best_result = col[:]
                break
    
    # Fallback: return the seed coloring
    if best_result is None:
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    return best_result
# EVOLVE-BLOCK-END
