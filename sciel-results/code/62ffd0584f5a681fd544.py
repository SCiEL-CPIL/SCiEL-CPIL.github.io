import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin (evaluator kills at 150s)

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

    # Precompute K4 edge index arrays
    k4_edge_idxs = []
    for k4e in k4s:
        k4_edge_idxs.append(np.array([edge_idx[e] for e in k4e], dtype=np.int32))

    # Convert edge_to_k4s to numpy arrays
    edge_to_k4s_np = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]

    def count_colors(coloring):
        return len(np.unique(coloring))

    def check_valid(coloring):
        """Check all K4s have at least 5 distinct colors using bitmask."""
        for idxs in k4_edge_idxs:
            mask = 0
            for idx in idxs:
                mask |= (1 << int(coloring[idx]))
            if bin(mask).count('1') < 5:
                return False
        return True

    def count_violations(coloring):
        violations = 0
        for idxs in k4_edge_idxs:
            mask = 0
            for idx in idxs:
                mask |= (1 << int(coloring[idx]))
            if bin(mask).count('1') < 5:
                violations += 1
        return violations

    def check_violations_subset(coloring, k4_indices):
        violations = 0
        for k4_i in k4_indices:
            idxs = k4_edge_idxs[k4_i]
            mask = 0
            for idx in idxs:
                mask |= (1 << int(coloring[idx]))
            if bin(mask).count('1') < 5:
                violations += 1
        return violations

    random.seed(42)
    
    # Generate candidate colorings from structural approaches
    candidates = []
    
    # Seed coloring
    seed_coloring = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    if check_valid(seed_coloring):
        candidates.append(seed_coloring.copy())
    
    # Try various structural colorings
    for mod in range(13, 25):
        if time.time() > deadline - 60:
            break
        
        # XOR-based
        for trial in range(8):
            if time.time() > deadline - 60:
                break
            labels = np.random.randint(0, mod, size=n)
            coloring = np.zeros(m, dtype=np.int32)
            for idx, (i, j) in enumerate(edges):
                coloring[idx] = (labels[i] ^ labels[j]) % mod
            if check_valid(coloring):
                candidates.append(coloring.copy())
        
        # Sum-based
        for trial in range(8):
            if time.time() > deadline - 60:
                break
            labels = np.random.randint(0, mod, size=n)
            coloring = np.zeros(m, dtype=np.int32)
            for idx, (i, j) in enumerate(edges):
                coloring[idx] = (labels[i] + labels[j]) % mod
            if check_valid(coloring):
                candidates.append(coloring.copy())
        
        # Difference-based
        for trial in range(8):
            if time.time() > deadline - 60:
                break
            labels = np.random.randint(0, mod, size=n)
            coloring = np.zeros(m, dtype=np.int32)
            for idx, (i, j) in enumerate(edges):
                coloring[idx] = abs(labels[i] - labels[j]) % mod
            if check_valid(coloring):
                candidates.append(coloring.copy())
    
    # Sort by number of colors
    candidates.sort(key=lambda c: count_colors(c))
    
    # If no valid candidates, use seed
    if not candidates:
        candidates.append(seed_coloring.copy())
    
    best_coloring = candidates[0].copy()
    best_colors = count_colors(best_coloring)
    
    # Local search to reduce colors
    for start_coloring in candidates:
        if time.time() > deadline - 15:
            break
        
        coloring = start_coloring.copy()
        current_colors = count_colors(coloring)
        
        if current_colors >= best_colors:
            continue
        
        max_iters = 20000
        for iter_count in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Try to eliminate the least frequently used color
            unique, counts = np.unique(coloring, return_counts=True)
            if len(unique) == 0:
                break
            
            target_color = unique[np.argmin(counts)]
            target_edges = np.where(coloring == target_color)[0]
            
            if len(target_edges) == 0:
                break
            
            # Try recoloring each edge of target_color to another color
            success = True
            old_colors = {}
            
            for ei in target_edges:
                ei = int(ei)
                old_colors[ei] = int(coloring[ei])
                
                found = False
                for new_color in unique:
                    if new_color == target_color:
                        continue
                    
                    coloring[ei] = new_color
                    affected_k4s = edge_to_k4s_np[ei]
                    if check_violations_subset(coloring, affected_k4s) == 0:
                        found = True
                        break
                    else:
                        coloring[ei] = old_colors[ei]
                
                if not found:
                    # Restore all changes
                    for ri, rc in old_colors.items():
                        coloring[ri] = rc
                    success = False
                    break
            
            if success:
                if check_valid(coloring):
                    new_count = count_colors(coloring)
                    if new_count < current_colors:
                        current_colors = new_count
                        best_colors = new_count
                        best_coloring = coloring.copy()
                    else:
                        # Restore
                        for ri, rc in old_colors.items():
                            coloring[ri] = rc
            
            if best_colors <= 13:
                break
        
        if best_colors <= 13:
            break
    
    # Final polish: try single-edge recoloring to reduce colors further
    if best_colors > 13 and time.time() < deadline - 5:
        coloring = best_coloring.copy()
        current_colors = best_colors
        
        for _ in range(10000):
            if time.time() > deadline - 3:
                break
            
            unique, counts = np.unique(coloring, return_counts=True)
            if len(unique) == 0:
                break
            
            # Try to recolor a random edge from the least common color
            target_color = int(unique[np.argmin(counts)])
            target_edges = np.where(coloring == target_color)[0]
            
            if len(target_edges) == 0:
                break
            
            idx = int(random.choice(target_edges))
            old_color = int(coloring[idx])
            
            for new_color in unique:
                if new_color == old_color:
                    continue
                
                coloring[idx] = new_color
                if check_valid(coloring):
                    new_count = count_colors(coloring)
                    if new_count < current_colors:
                        current_colors = new_count
                        best_colors = new_count
                        best_coloring = coloring.copy()
                        break
                    else:
                        coloring[idx] = old_color
                        break
                else:
                    coloring[idx] = old_color
            
            if best_colors <= 13:
                break
    
    # Return the best coloring found
    result = [int(c) for c in best_coloring]
    # Remap colors to be contiguous starting from 0
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
