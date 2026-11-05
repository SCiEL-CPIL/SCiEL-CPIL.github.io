import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120
    
    # Precompute K4 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(k4_edges)
    
    # Edge index map
    edge_idx = {}
    for idx, e in enumerate(edges):
        edge_idx[e] = idx
    
    # Precompute which K4s each edge belongs to
    edge_to_k4s = [[] for _ in range(m)]
    for k4_i, k4e in enumerate(k4s):
        for e in k4e:
            edge_to_k4s[edge_idx[e]].append(k4_i)
    
    # Precompute K4 edge index arrays for fast access
    k4_edge_idxs = []
    for k4e in k4s:
        k4_edge_idxs.append([edge_idx[e] for e in k4e])
    
    # Precompute K4 edge index arrays as numpy arrays for vectorized ops
    k4_edge_idxs_np = np.array(k4_edge_idxs, dtype=np.int32)
    
    def count_colors(coloring):
        return len(np.unique(coloring))
    
    def count_violations(coloring):
        """Count K4s with fewer than 5 distinct colors using vectorized ops."""
        # coloring is shape (120,), k4_edge_idxs_np is shape (1820, 6)
        k4_colors = coloring[k4_edge_idxs_np]  # shape (1820, 6)
        # For each K4, count distinct colors
        # Sort each row and count unique elements
        sorted_k4 = np.sort(k4_colors, axis=1)
        # Count distinct: compare adjacent elements
        diffs = np.diff(sorted_k4, axis=1)
        distinct_counts = 1 + np.sum(diffs != 0, axis=1)
        violations = np.sum(distinct_counts < 5)
        return int(violations)
    
    def check_valid(coloring):
        return count_violations(coloring) == 0
    
    def check_violations_subset(coloring, k4_indices):
        """Check violations only for a subset of K4s."""
        if len(k4_indices) == 0:
            return 0
        k4_idxs = k4_edge_idxs_np[k4_indices]
        k4_colors = coloring[k4_idxs]
        sorted_k4 = np.sort(k4_colors, axis=1)
        diffs = np.diff(sorted_k4, axis=1)
        distinct_counts = 1 + np.sum(diffs != 0, axis=1)
        return int(np.sum(distinct_counts < 5))
    
    # Build 1-factorization of K16
    # K16 has 15 perfect matchings, each with 8 edges
    # Standard 1-factorization using cyclic method
    vertices = list(range(15)) + [15]  # 15 vertices + infinity
    factorization = []
    for round_ in range(15):
        matching = []
        # Fix vertex 15, rotate others
        for k in range(7):
            v1 = (round_ + k) % 15
            v2 = (round_ - k - 1) % 15
            matching.append((v1, v2))
        matching.append((15, (round_ + 7) % 15))
        factorization.append(matching)
    
    # Verify factorization covers all edges
    all_edges_in_factorization = set()
    for matching in factorization:
        for e in matching:
            all_edges_in_factorization.add(tuple(sorted(e)))
    assert len(all_edges_in_factorization) == 120, "Factorization incomplete"
    
    # Create 1-factorization based coloring: each matching gets a base color
    # Then perturb with secondary offsets
    def make_factorization_coloring(base_colors, perturbations):
        """
        base_colors: list of 15 colors for the 15 matchings
        perturbations: list of 120 perturbations (0 or small offset)
        """
        coloring = np.zeros(m, dtype=int)
        for round_ in range(15):
            base = base_colors[round_]
            for e in factorization[round_]:
                idx = edge_idx[tuple(sorted(e))]
                coloring[idx] = base + perturbations[idx]
        return coloring
    
    # Generate candidate colorings
    structural_colorings = []
    
    # Try 1-factorization with 13 base colors (some matchings share colors)
    # and perturbations to break K4 color collisions
    random.seed(42)
    
    # Strategy: assign base colors from a small palette, then add perturbations
    for num_base in range(13, 17):
        if time.time() > deadline - 40:
            break
        for trial in range(50):
            if time.time() > deadline - 40:
                break
            
            # Assign base colors to 15 matchings using num_base colors
            base_colors = [random.randint(0, num_base - 1) for _ in range(15)]
            
            # Generate perturbations: small random offsets
            # Perturbation should be small enough to not create too many new colors
            # but large enough to break K4 collisions
            perturbations = np.zeros(m, dtype=int)
            for idx in range(m):
                if random.random() < 0.3:  # 30% of edges get perturbed
                    perturbations[idx] = random.randint(1, 3)
            
            coloring_struct = make_factorization_coloring(base_colors, perturbations)
            
            # Check validity
            if check_valid(coloring_struct):
                structural_colorings.append(coloring_struct.copy())
    
    # Also try the original seed coloring
    seed_coloring = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=int)
    if check_valid(seed_coloring):
        structural_colorings.append(seed_coloring.copy())
    
    # Try XOR-based with vertex labels
    for mod in [13, 14, 15, 16, 17]:
        if time.time() > deadline - 35:
            break
        for trial in range(30):
            if time.time() > deadline - 35:
                break
            labels = [random.randint(0, mod - 1) for _ in range(n)]
            coloring_struct = np.zeros(m, dtype=int)
            for idx, (i, j) in enumerate(edges):
                coloring_struct[idx] = (labels[i] ^ labels[j]) % mod
            if check_valid(coloring_struct):
                structural_colorings.append(coloring_struct.copy())
    
    # Try sum-based with vertex labels
    for mod in [13, 14, 15, 16, 17]:
        if time.time() > deadline - 30:
            break
        for trial in range(30):
            if time.time() > deadline - 30:
                break
            labels = [random.randint(0, mod - 1) for _ in range(n)]
            coloring_struct = np.zeros(m, dtype=int)
            for idx, (i, j) in enumerate(edges):
                coloring_struct[idx] = (labels[i] + labels[j]) % mod
            if check_valid(coloring_struct):
                structural_colorings.append(coloring_struct.copy())
    
    # Sort structural colorings by number of colors
    structural_colorings.sort(key=lambda c: count_colors(c))
    
    best_coloring = None
    best_colors = 120  # worst case
    
    # If we found valid structural colorings, use the best one
    if structural_colorings:
        best_coloring = structural_colorings[0].copy()
        best_colors = count_colors(best_coloring)
    else:
        # Fallback: use seed if valid
        if check_valid(seed_coloring):
            best_coloring = seed_coloring.copy()
            best_colors = count_colors(best_coloring)
        else:
            # Emergency: create a valid coloring by using many colors
            # Assign each edge a unique color, then merge
            coloring_emergency = np.array([i for i in range(m)], dtype=int)
            best_coloring = coloring_emergency.copy()
            best_colors = m
    
    # Phase 2: Local search from each structural starting point
    for start_coloring in structural_colorings:
        if time.time() > deadline - 15:
            break
        
        coloring = start_coloring.copy()
        current_colors = count_colors(coloring)
        
        if current_colors >= best_colors:
            continue
        
        # Local search: try to reduce colors
        max_iters = 50000
        iter_count = 0
        no_improve_count = 0
        max_no_improve = 5000
        
        while iter_count < max_iters:
            if time.time() > deadline - 5:
                break
            
            iter_count += 1
            
            # Find the most frequently used color
            unique, counts = np.unique(coloring, return_counts=True)
            if len(unique) == 0:
                break
            
            target_color = unique[np.argmax(counts)]
            
            # Find edges with this color
            target_edges = np.where(coloring == target_color)[0]
            
            if len(target_edges) == 0:
                break
            
            # Try to recolor a random edge with target_color to a different color
            idx = int(random.choice(target_edges))
            old_color = int(coloring[idx])
            
            # Get all colors currently in use
            all_colors = list(unique)
            random.shuffle(all_colors)
            
            improved = False
            for new_color in all_colors:
                if new_color == old_color:
                    continue
                
                coloring[idx] = new_color
                
                # Quick check: only need to check K4s containing this edge
                affected_k4s = edge_to_k4s[idx]
                if check_violations_subset(coloring, affected_k4s) == 0:
                    # Full check to be safe
                    if check_valid(coloring):
                        new_count = count_colors(coloring)
                        if new_count < current_colors:
                            current_colors = new_count
                            if new_count < best_colors:
                                best_colors = new_count
                                best_coloring = coloring.copy()
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            coloring[idx] = old_color
                            break
                    else:
                        coloring[idx] = old_color
                        break
                else:
                    coloring[idx] = old_color
            
            if not improved:
                no_improve_count += 1
                # Try random swaps
                if random.random() < 0.3:
                    i1 = random.randint(0, m - 1)
                    i2 = random.randint(0, m - 1)
                    if i1 != i2:
                        c1, c2 = int(coloring[i1]), int(coloring[i2])
                        if c1 != c2:
                            coloring[i1], coloring[i2] = c2, c1
                            if check_valid(coloring):
                                new_count = count_colors(coloring)
                                if new_count < current_colors:
                                    current_colors = new_count
                                    if new_count < best_colors:
                                        best_colors = new_count
                                        best_coloring = coloring.copy()
                                    no_improve_count = 0
                                else:
                                    coloring[i1], coloring[i2] = c1, c2
                            else:
                                coloring[i1], coloring[i2] = c1, c2
            
            # If we've reached the theoretical minimum, stop
            if best_colors <= 13:
                break
            
            # Restart if stuck
            if no_improve_count > max_no_improve:
                break
    
    # Phase 3: Final polish - try to reduce colors further from the best coloring
    if best_colors > 13 and best_coloring is not None:
        coloring = best_coloring.copy()
        current_colors = best_colors
        
        for _ in range(30000):
            if time.time() > deadline - 2:
                break
            
            # Find the most frequently used color
            unique, counts = np.unique(coloring, return_counts=True)
            if len(unique) == 0:
                break
            
            target_color = unique[np.argmax(counts)]
            
            # Find edges with this color
            target_edges = np.where(coloring == target_color)[0]
            
            if len(target_edges) == 0:
                break
            
            # Try to recolor a random edge with target_color to a different color
            idx = int(random.choice(target_edges))
            old_color = int(coloring[idx])
            
            # Get all colors currently in use
            all_colors = list(unique)
            random.shuffle(all_colors)
            
            for new_color in all_colors:
                if new_color == old_color:
                    continue
                
                coloring[idx] = new_color
                
                # Quick check: only need to check K4s containing this edge
                affected_k4s = edge_to_k4s[idx]
                if check_violations_subset(coloring, affected_k4s) == 0:
                    if check_valid(coloring):
                        new_count = count_colors(coloring)
                        if new_count < current_colors:
                            current_colors = new_count
                            if new_count < best_colors:
                                best_colors = new_count
                                best_coloring = coloring.copy()
                            break
                        else:
                            coloring[idx] = old_color
                            break
                    else:
                        coloring[idx] = old_color
                        break
                else:
                    coloring[idx] = old_color
            
            if best_colors <= 13:
                break
    
    # Return the best coloring found
    if best_coloring is None:
        # Emergency fallback
        result = list(range(m))
    else:
        result = [int(c) for c in best_coloring]
    
    # Ensure colors are in valid range and remap to minimize
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
