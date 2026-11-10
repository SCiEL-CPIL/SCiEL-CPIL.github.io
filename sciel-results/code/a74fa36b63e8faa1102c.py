"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import Counter

def construct():
    """
    Construct an edge-coloring of K16 minimizing colors used
    such that every K4 has at least 5 distinct colors.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute all K4s and their edge indices
    k4_edges = []
    for quad in combinations(range(n), 4):
        edges = []
        for a, b in combinations(quad, 2):
            idx = a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)
            edges.append(idx)
        k4_edges.append(edges)
    
    num_k4s = len(k4_edges)
    
    # Precompute which K4s contain each edge for fast validity checking
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
    
    # Helper to check validity of a coloring using precomputed K4-edge mapping
    def is_valid(coloring):
        for edges in k4_edges:
            colors_in_k4 = set(coloring[e] for e in edges)
            if len(colors_in_k4) < 5:
                return False
        return True
    
    # Helper to check if changing edge_idx to new_color keeps validity
    # Only checks K4s that contain edge_idx
    def is_valid_after_change(coloring, edge_idx, new_color):
        for k4_idx in edge_to_k4s[edge_idx]:
            edges = k4_edges[k4_idx]
            colors_in_k4 = set()
            for e in edges:
                if e == edge_idx:
                    colors_in_k4.add(new_color)
                else:
                    colors_in_k4.add(coloring[e])
            if len(colors_in_k4) < 5:
                return False
        return True
    
    # Helper to count distinct colors used
    def count_colors(coloring):
        return len(set(coloring))
    
    # Start with the seed solution: (i+j) % 17
    def seed_coloring():
        return np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    best_coloring = seed_coloring()
    best_score = count_colors(best_coloring)  # 17
    
    # Try to improve by reducing colors
    # Strategy: try to merge colors one by one
    
    def try_merge_colors(coloring, max_colors, max_iters=2000):
        """Try to reduce number of colors by merging."""
        coloring = coloring.copy()
        current_colors = sorted(set(coloring))
        
        if len(current_colors) <= max_colors:
            return coloring, len(current_colors)
        
        for _ in range(max_iters):
            current_colors = sorted(set(coloring))
            if len(current_colors) <= max_colors:
                break
            
            # Try to merge the two least frequent colors
            color_counts = Counter(coloring)
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
            
            merged = False
            for c1, count1 in sorted_colors[:6]:
                for c2, count2 in sorted_colors[6:12]:
                    if c1 == c2:
                        continue
                    
                    # Try merging c2 into c1
                    test_coloring = coloring.copy()
                    test_coloring[test_coloring == c2] = c1
                    
                    if is_valid(test_coloring):
                        coloring = test_coloring
                        merged = True
                        break
                if merged:
                    break
            
            if not merged:
                break
        
        return coloring, count_colors(coloring)
    
    # Start with seed and try to reduce
    current = best_coloring.copy()
    
    # Try reducing from 17 down
    for target in range(best_score - 1, 12, -1):  # Try 16, 15, 14, 13
        if time.time() > deadline - 40:
            break
        
        # Reset to best known
        current = best_coloring.copy()
        
        # Try to merge colors to reach target
        improved, score = try_merge_colors(current, target, max_iters=1000)
        
        if score < best_score:
            best_coloring = improved
            best_score = score
    
    # If we have time, try local search with targeted color reduction
    if time.time() < deadline - 10 and best_score > 13:
        rng = random.Random(42)
        current = best_coloring.copy()
        current_score = best_score
        
        # First, try to systematically reduce by targeting specific colors
        # For each color, try to reassign its edges to other colors
        for target_color in range(current_score, 0, -1):
            if time.time() > deadline - 5:
                break
            
            # Find all edges with this color
            edges_with_color = np.where(current == target_color)[0]
            if len(edges_with_color) == 0:
                continue
            
            # Try to reassign each edge to a different color
            for edge_idx in edges_with_color:
                if time.time() > deadline - 5:
                    break
                
                old_color = current[edge_idx]
                if old_color != target_color:
                    continue
                
                # Try assigning to other colors
                for new_color in range(target_color):
                    if is_valid_after_change(current, edge_idx, new_color):
                        current[edge_idx] = new_color
                        break
            
            # Check if we reduced the score
            new_score = count_colors(current)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_coloring = current.copy()
                    best_score = current_score
        
        # Additional random local search
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            
            # Pick a random edge and try a random color
            edge_idx = rng.randint(0, num_edges - 1)
            old_color = current[edge_idx]
            new_color = rng.randint(0, current_score - 1)
            
            if new_color == old_color:
                continue
            
            current[edge_idx] = new_color
            
            if is_valid_after_change(current, edge_idx, new_color):
                new_score = count_colors(current)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_coloring = current.copy()
                        best_score = current_score
            else:
                current[edge_idx] = old_color
    
    # Final check
    if not is_valid(best_coloring):
        best_coloring = seed_coloring()
        best_score = count_colors(best_coloring)
    
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
