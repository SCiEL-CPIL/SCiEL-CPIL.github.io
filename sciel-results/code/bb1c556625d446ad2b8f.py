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
        k4_colors = coloring[k4_edge_idxs_np]
        sorted_k4 = np.sort(k4_colors, axis=1)
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
    
    def generate_vertex_label_coloring(labels, mod, op_type='add'):
        """Generate edge coloring from vertex labels using a binary operation."""
        coloring = np.zeros(m, dtype=int)
        for idx, (i, j) in enumerate(edges):
            if op_type == 'add':
                coloring[idx] = (labels[i] + labels[j]) % mod
            elif op_type == 'xor':
                coloring[idx] = (labels[i] ^ labels[j]) % mod
            elif op_type == 'mul':
                coloring[idx] = (labels[i] * labels[j]) % mod
            elif op_type == 'diff':
                coloring[idx] = abs(labels[i] - labels[j]) % mod
            elif op_type == 'add_mul':
                # Combination: a + b + a*b mod mod
                coloring[idx] = (labels[i] + labels[j] + labels[i] * labels[j]) % mod
            elif op_type == 'affine':
                # a * k + b mod mod for some fixed k
                k = 2  # fixed multiplier
                coloring[idx] = (k * labels[i] + labels[j]) % mod
        return coloring
    
    structural_colorings = []
    random.seed(42)
    
    # Try systematic vertex label constructions
    # For small moduli, try all possible label assignments (or a good subset)
    for mod in range(13, 20):
        if time.time() > deadline - 60:
            break
        
        # For mod=13,14,15: try structured labelings
        # Use labels that are spread out to maximize color diversity
        if mod <= 15:
            # Try arithmetic progression labels
            for start in range(mod):
                for step in range(1, mod):
                    if time.time() > deadline - 60:
                        break
                    labels = [(start + step * i) % mod for i in range(n)]
                    for op_type in ['add', 'xor', 'mul', 'diff', 'add_mul', 'affine']:
                        coloring_struct = generate_vertex_label_coloring(labels, mod, op_type)
                        if check_valid(coloring_struct):
                            structural_colorings.append(coloring_struct.copy())
            
            # Try random labelings
            for trial in range(100):
                if time.time() > deadline - 60:
                    break
                labels = [random.randint(0, mod - 1) for _ in range(n)]
                for op_type in ['add', 'xor', 'mul', 'diff', 'add_mul', 'affine']:
                    coloring_struct = generate_vertex_label_coloring(labels, mod, op_type)
                    if check_valid(coloring_struct):
                        structural_colorings.append(coloring_struct.copy())
        else:
            # For larger moduli, just try random
            for trial in range(50):
                if time.time() > deadline - 60:
                    break
                labels = [random.randint(0, mod - 1) for _ in range(n)]
                for op_type in ['add', 'xor', 'mul']:
                    coloring_struct = generate_vertex_label_coloring(labels, mod, op_type)
                    if check_valid(coloring_struct):
                        structural_colorings.append(coloring_struct.copy())
    
    # Also try the seed coloring
    seed_coloring = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=int)
    if check_valid(seed_coloring):
        structural_colorings.append(seed_coloring.copy())
    
    # Try 1-factorization based coloring
    vertices = list(range(15)) + [15]
    factorization = []
    for round_ in range(15):
        matching = []
        for k in range(7):
            v1 = (round_ + k) % 15
            v2 = (round_ - k - 1) % 15
            matching.append((v1, v2))
        matching.append((15, (round_ + 7) % 15))
        factorization.append(matching)
    
    for num_base in range(13, 17):
        if time.time() > deadline - 50:
            break
        for trial in range(30):
            if time.time() > deadline - 50:
                break
            base_colors = [random.randint(0, num_base - 1) for _ in range(15)]
            perturbations = np.zeros(m, dtype=int)
            for idx in range(m):
                if random.random() < 0.3:
                    perturbations[idx] = random.randint(1, 3)
            coloring_struct = np.zeros(m, dtype=int)
            for round_ in range(15):
                base = base_colors[round_]
                for e in factorization[round_]:
                    idx = edge_idx[tuple(sorted(e))]
                    coloring_struct[idx] = base + perturbations[idx]
            if check_valid(coloring_struct):
                structural_colorings.append(coloring_struct.copy())
    
    # Sort by number of colors
    structural_colorings.sort(key=lambda c: count_colors(c))
    
    best_coloring = None
    best_colors = 120
    
    if structural_colorings:
        best_coloring = structural_colorings[0].copy()
        best_colors = count_colors(best_coloring)
    else:
        # Emergency fallback
        best_coloring = np.array([i for i in range(m)], dtype=int)
        best_colors = m
    
    # Local search to reduce colors
    if best_coloring is not None:
        coloring = best_coloring.copy()
        current_colors = best_colors
        
        max_iters = 80000
        iter_count = 0
        no_improve = 0
        max_no_improve = 3000
        
        while iter_count < max_iters:
            if time.time() > deadline - 3:
                break
            iter_count += 1
            
            unique, counts = np.unique(coloring, return_counts=True)
            if len(unique) == 0:
                break
            
            # Target the most frequent color
            target_color = unique[np.argmax(counts)]
            target_edges = np.where(coloring == target_color)[0]
            
            if len(target_edges) == 0:
                break
            
            idx = int(random.choice(target_edges))
            old_color = int(coloring[idx])
            
            all_colors = list(unique)
            random.shuffle(all_colors)
            
            improved = False
            for new_color in all_colors:
                if new_color == old_color:
                    continue
                
                coloring[idx] = new_color
                affected_k4s = edge_to_k4s[idx]
                
                if check_violations_subset(coloring, affected_k4s) == 0:
                    if check_valid(coloring):
                        new_count = count_colors(coloring)
                        if new_count < current_colors:
                            current_colors = new_count
                            if new_count < best_colors:
                                best_colors = new_count
                                best_coloring = coloring.copy()
                            improved = True
                            no_improve = 0
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
                no_improve += 1
                # Occasional random perturbation
                if random.random() < 0.2:
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
                                    no_improve = 0
                                else:
                                    coloring[i1], coloring[i2] = c1, c2
                            else:
                                coloring[i1], coloring[i2] = c1, c2
            
            if best_colors <= 13:
                break
            if no_improve > max_no_improve:
                break
    
    result = [int(c) for c in best_coloring]
    
    # Remap colors to consecutive integers starting from 0
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
