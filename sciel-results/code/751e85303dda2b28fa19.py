"""Program for hypergraph Ramsey R(4,5;3) coloring task."""
import numpy as np
from itertools import combinations
import random
import time

def construct():
    """
    Construct a valid coloring of triples on n vertices with no all-red K4^(3)
    and no all-blue K5^(3). Maximize n.
    
    Strategy: Use a randomized local search / simulated annealing approach.
    Start with a random coloring for n=34 (or smaller if needed), then iteratively
    flip triple colors to eliminate violations (all-red K4s and all-blue K5s).
    """
    # We aim for n=34, the known published construction size
    target_n = 34
    
    # Precompute all triples in lexicographic order
    all_triples = list(combinations(range(target_n), 3))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {}
    for i, (a, b, c) in enumerate(all_triples):
        triple_to_idx[(a, b, c)] = i
    
    # Precompute for each 4-set, which 4 triples it contains
    # A K4^(3) has C(4,3)=4 triples
    k4_sets = []
    for quad in combinations(range(target_n), 4):
        triples_in_quad = []
        for tri in combinations(quad, 3):
            triples_in_quad.append(triple_to_idx[tri])
        k4_sets.append(triples_in_quad)
    
    # Precompute for each 5-set, which 10 triples it contains
    # A K5^(3) has C(5,3)=10 triples
    k5_sets = []
    for quint in combinations(range(target_n), 5):
        triples_in_quint = []
        for tri in combinations(quint, 3):
            triples_in_quint.append(triple_to_idx[tri])
        k5_sets.append(triples_in_quint)
    
    # Initialize with a random coloring
    colors = np.random.randint(0, 2, size=num_triples, dtype=np.uint8)
    
    # Helper: check if a 4-set is all-red (all 4 triples are color 0)
    def is_all_red_k4(triple_indices):
        return all(colors[i] == 0 for i in triple_indices)
    
    # Helper: check if a 5-set is all-blue (all 10 triples are color 1)
    def is_all_blue_k5(triple_indices):
        return all(colors[i] == 1 for i in triple_indices)
    
    # Count violations
    def count_violations():
        red_viol = 0
        blue_viol = 0
        for tri_indices in k4_sets:
            if is_all_red_k4(tri_indices):
                red_viol += 1
        for tri_indices in k5_sets:
            if is_all_blue_k5(triple_indices):
                blue_viol += 1
        return red_viol + blue_viol
    
    # Local search: try to flip triples to reduce violations
    max_iterations = 500000
    current_viol = count_violations()
    
    if current_viol == 0:
        return {'n': target_n, 'colors': colors.tolist()}
    
    # Use a priority-based approach: for each violating structure, try flipping one of its triples
    # We'll use a simple hill-climbing with random restarts
    
    best_colors = colors.copy()
    best_viol = current_viol
    
    for restart in range(20):
        # Random restart
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.uint8)
        current_viol = count_violations()
        
        if current_viol < best_viol:
            best_colors = colors.copy()
            best_viol = current_viol
        
        if best_viol == 0:
            break
        
        # Local search phase
        no_improve_count = 0
        for it in range(max_iterations // 20):
            if current_viol == 0:
                break
            
            # Pick a random violating structure and flip one of its triples
            # Find a random K4 that is all-red
            found_flip = False
            for _ in range(10):  # Try a few times to find a violating structure
                # Randomly pick a K4
                idx = random.randint(0, len(k4_sets) - 1)
                tri_indices = k4_sets[idx]
                if is_all_red_k4(tri_indices):
                    # Flip a random triple in this K4 to blue
                    flip_idx = random.choice(tri_indices)
                    colors[flip_idx] = 1
                    found_flip = True
                    break
                # Randomly pick a K5
                idx = random.randint(0, len(k5_sets) - 1)
                tri_indices = k5_sets[idx]
                if is_all_blue_k5(tri_indices):
                    # Flip a random triple in this K5 to red
                    flip_idx = random.choice(tri_indices)
                    colors[flip_idx] = 0
                    found_flip = True
                    break
            
            if not found_flip:
                # No easy violation found, do a random flip
                flip_idx = random.randint(0, num_triples - 1)
                colors[flip_idx] = 1 - colors[flip_idx]
            
            new_viol = count_violations()
            
            if new_viol < current_viol:
                current_viol = new_viol
                no_improve_count = 0
                if current_viol < best_viol:
                    best_colors = colors.copy()
                    best_viol = current_viol
                    if best_viol == 0:
                        break
            elif new_viol == current_viol:
                # Accept with some probability (plateau)
                if random.random() < 0.3:
                    current_viol = new_viol
                    no_improve_count += 1
                else:
                    # Undo the flip
                    colors[flip_idx] = 1 - colors[flip_idx]
                    no_improve_count += 1
            else:
                # Undo the flip
                colors[flip_idx] = 1 - colors[flip_idx]
                no_improve_count += 1
            
            if no_improve_count > 5000:
                break
        
        if best_viol == 0:
            break
    
    # Use the best coloring found
    final_colors = best_colors
    final_viol = best_viol
    
    if final_viol == 0:
        return {'n': target_n, 'colors': final_colors.tolist()}
    
    # If we couldn't find a valid coloring for n=34, try smaller n
    # Try n=33, 32, etc.
    for n in range(target_n - 1, 7, -1):
        all_triples_n = list(combinations(range(n), 3))
        num_triples_n = len(all_triples_n)
        
        triple_to_idx_n = {}
        for i, (a, b, c) in enumerate(all_triples_n):
            triple_to_idx_n[(a, b, c)] = i
        
        k4_sets_n = []
        for quad in combinations(range(n), 4):
            triples_in_quad = []
            for tri in combinations(quad, 3):
                triples_in_quad.append(triple_to_idx_n[tri])
            k4_sets_n.append(triples_in_quad)
        
        k5_sets_n = []
        for quint in combinations(range(n), 5):
            triples_in_quint = []
            for tri in combinations(quint, 3):
                triples_in_quint.append(triple_to_idx_n[tri])
            k5_sets_n.append(triples_in_quint)
        
        # Try random coloring with local search
        for attempt in range(10):
            colors_n = np.random.randint(0, 2, size=num_triples_n, dtype=np.uint8)
            
            def is_all_red_k4_n(triple_indices):
                return all(colors_n[i] == 0 for i in triple_indices)
            
            def is_all_blue_k5_n(triple_indices):
                return all(colors_n[i] == 1 for i in triple_indices)
            
            def count_violations_n():
                red_viol = 0
                blue_viol = 0
                for tri_indices in k4_sets_n:
                    if is_all_red_k4_n(tri_indices):
                        red_viol += 1
                for tri_indices in k5_sets_n:
                    if is_all_blue_k5_n(triple_indices):
                        blue_viol += 1
                return red_viol + blue_viol
            
            current_viol_n = count_violations_n()
            
            if current_viol_n == 0:
                return {'n': n, 'colors': colors_n.tolist()}
            
            # Local search
            for it in range(100000):
                if current_viol_n == 0:
                    break
                
                found_flip_n = False
                for _ in range(5):
                    idx = random.randint(0, len(k4_sets_n) - 1)
                    tri_indices = k4_sets_n[idx]
                    if is_all_red_k4_n(tri_indices):
                        flip_idx = random.choice(tri_indices)
                        colors_n[flip_idx] = 1
                        found_flip_n = True
                        break
                    idx = random.randint(0, len(k5_sets_n) - 1)
                    tri_indices = k5_sets_n[idx]
                    if is_all_blue_k5_n(tri_indices):
                        flip_idx = random.choice(tri_indices)
                        colors_n[flip_idx] = 0
                        found_flip_n = True
                        break
                
                if not found_flip_n:
                    flip_idx = random.randint(0, num_triples_n - 1)
                    colors_n[flip_idx] = 1 - colors_n[flip_idx]
                
                new_viol_n = count_violations_n()
                
                if new_viol_n <= current_viol_n:
                    current_viol_n = new_viol_n
                else:
                    colors_n[flip_idx] = 1 - colors_n[flip_idx]
            
            if count_violations_n() == 0:
                return {'n': n, 'colors': colors_n.tolist()}
    
    # Fallback: return the best we found or a small valid coloring
    # For n=9, use the modular coloring which is known to be valid
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
