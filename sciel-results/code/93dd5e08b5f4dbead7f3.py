"""Improved program for kissing number in dimension 11 using W2+W3 vectors."""
import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate candidates: Weight 2 and Weight 3 vectors
    candidates = []
    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[positions[0]] = signs[0]
            v[positions[1]] = signs[1]
            candidates.append(v)
            
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # 2. Compute dot products and conflicts
    # Conflict if dot_product >= 2 (since norms are 2 and 3, max norm squared is 3)
    # Condition: ||u-v||^2 >= max_norm_sq = 3
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # For W2 (norm^2=2) and W3 (norm^2=3): 2+3-2u.v >= 3 => 2-2u.v >= 0 => u.v <= 1
    # For W3 (norm^2=3) and W3 (norm^2=3): 3+3-2u.v >= 3 => 3-2u.v >= 0 => u.v <= 1.5 => u.v <= 1
    # For W2 and W2: 2+2-2u.v >= 3 => 1-2u.v >= 0 => u.v <= 0.5 => u.v <= 0
    # Wait! W2-W2 conflict condition is u.v <= 0.
    # Let's re-verify.
    # If we mix W2 and W3, max norm squared is 3.
    # Pair (W2, W2): dist^2 >= 3. 2+2-2u.v >= 3 => 2u.v <= 1 => u.v <= 0.
    # Pair (W2, W3): dist^2 >= 3. 2+3-2u.v >= 3 => 2u.v <= 2 => u.v <= 1.
    # Pair (W3, W3): dist^2 >= 3. 3+3-2u.v >= 3 => 2u.v <= 3 => u.v <= 1.
    
    # So the conflict conditions are different!
    # We cannot just use a single threshold.
    # We need to check each pair based on their weights.
    
    norms_sq = np.zeros(n, dtype=np.int64)
    for i in range(n):
        norms_sq[i] = np.sum(candidates[i]**2)
        
    dots = np.dot(candidates, candidates.T)
    
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*dots[i, j]
    # Conflict if dist_sq[i, j] < 3
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 3
    np.fill_diagonal(conflicts, False)
    
    # 3. Greedy Independent Set with Local Search
    # Convert conflicts to adjacency list for faster neighbor checks
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())
        
    def greedy_min_degree():
        available = set(range(n))
        selected = []
        
        while available:
            if time.time() > deadline:
                break
            # Pick node with min degree in available subgraph
            # Optimization: if available is large, sample
            if len(available) > 500:
                sample = random.sample(list(available), 500)
            else:
                sample = list(available)
                
            best_v = None
            min_deg = float('inf')
            for v in sample:
                deg = len(neighbors[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    
            if best_v is None:
                break
                
            selected.append(best_v)
            available.discard(best_v)
            for nb in neighbors[best_v]:
                if nb in available:
                    available.discard(nb)
                    
        return selected

    # Initial greedy
    np.random.seed(42)
    random.seed(42)
    
    best_selected = []
    
    # Run multiple greedy trials
    for trial in range(20):
        if time.time() > deadline - 20.0:
            break
        sel = greedy_min_degree()
        if len(sel) > len(best_selected):
            best_selected = sel
            
    # Local Search Improvement
    if best_selected and time.time() < deadline - 5.0:
        selected_set = set(best_selected)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # 1. Try to add any unselected node that doesn't conflict
            for v in range(n):
                if time.time() > deadline - 2.0:
                    break
                if v not in selected_set:
                    if not (neighbors[v] & selected_set):
                        selected_set.add(v)
                        improved = True
                        
            # 2. Try swaps: remove one, add others
            if not improved:
                # Randomly shuffle selected to avoid bias
                sel_list = list(selected_set)
                random.shuffle(sel_list)
                for v in sel_list:
                    if time.time() > deadline - 2.0:
                        break
                    # Nodes blocked by v
                    blocked = neighbors[v] - selected_set
                    if not blocked:
                        continue
                    
                    test_set = selected_set - {v}
                    added = 0
                    for b in blocked:
                        if b not in test_set and not (neighbors[b] & test_set):
                            test_set.add(b)
                            added += 1
                            
                    if added > 1: # Net gain
                        selected_set = test_set
                        improved = True
                        
    if not best_selected:
        # Fallback: just return one vector
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
        
    return candidates[list(best_selected)]
