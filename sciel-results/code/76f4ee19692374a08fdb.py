import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 148.0  # safety margin (evaluator kills at 150s)
    
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
        k4_edge_idxs.append(np.array([edge_idx[e] for e in k4e], dtype=np.int32))
    
    # Convert edge_to_k4s to numpy arrays for faster access
    edge_to_k4s_np = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]
    
    def count_colors(coloring):
        return len(np.unique(coloring))
    
    def count_violations(coloring):
        """Count K4s with fewer than 5 distinct colors."""
        violations = 0
        for idxs in k4_edge_idxs:
            colors = np.unique(coloring[idxs])
            if len(colors) < 5:
                violations += 1
        return violations
    
    def check_valid(coloring):
        return count_violations(coloring) == 0
    
    def check_violations_subset(coloring, k4_indices):
        """Check violations only for a subset of K4s."""
        violations = 0
        for k4_i in k4_indices:
            idxs = k4_edge_idxs[k4_i]
            colors = np.unique(coloring[idxs])
            if len(colors) < 5:
                violations += 1
        return violations
    
    # Start with the seed coloring
    coloring = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    best_coloring = coloring.copy()
    best_colors = count_colors(best_coloring)  # 17
    
    random.seed(42)
    
    # Phase 1: Try vertex-labeling structural approaches to find good starting points
    structural_colorings = []
    
    # Try XOR-based with vertex labels
    for mod in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
        if time.time() > deadline - 40:
            break
        for trial in range(20):
            if time.time() > deadline - 40:
                break
            labels = np.array([random.randint(0, mod - 1) for _ in range(n)], dtype=np.int32)
            coloring_struct = np.zeros(m, dtype=np.int32)
            for idx, (i, j) in enumerate(edges):
                coloring_struct[idx] = (labels[i] ^ labels[j]) % mod
            if check_valid(coloring_struct):
                structural_colorings.append(coloring_struct.copy())
    
    # Try sum-based with vertex labels
    for mod in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
        if time.time() > deadline - 35:
            break
        for trial in range(20):
            if time.time() > deadline - 35:
                break
            labels = np.array([random.randint(0, mod - 1) for _ in range(n)], dtype=np.int32)
            coloring_struct = np.zeros(m, dtype=np.int32)
            for idx, (i, j) in enumerate(edges):
                coloring_struct[idx] = (labels[i] + labels[j]) % mod
            if check_valid(coloring_struct):
                structural_colorings.append(coloring_struct.copy())
    
    # Try product-based with vertex labels
    for mod in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
        if time.time() > deadline - 30:
            break
        for trial in range(15):
            if time.time() > deadline - 30:
                break
            labels = np.array([random.randint(0, mod - 1) for _ in range(n)], dtype=np.int32)
            coloring_struct = np.zeros(m, dtype=np.int32)
            for idx, (i, j) in enumerate(edges):
                coloring_struct[idx] = (labels[i] * labels[j]) % mod
            if check_valid(coloring_struct):
                structural_colorings.append(coloring_struct.copy())
    
    # Add the seed coloring if it's valid
    if check_valid(best_coloring):
        structural_colorings.append(best_coloring.copy())
    
    # Sort structural colorings by number of colors
    structural_colorings.sort(key=lambda c: count_colors(c))
    
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
        
        while iter_count < max_iters:
            if time.time() > deadline - 10:
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
                affected_k4s = edge_to_k4s_np[idx]
                if check_violations_subset(coloring, affected_k4s) == 0:
                    # Full check to be safe
                    if check_valid(coloring):
                        new_count = count_colors(coloring)
                        if new_count < current_colors:
                            current_colors = new_count
                            best_colors = new_count
                            best_coloring = coloring.copy()
                            improved = True
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
                # Try random swaps
                if random.random() < 0.5:
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
                                    best_colors = new_count
                                    best_coloring = coloring.copy()
                                else:
                                    coloring[i1], coloring[i2] = c1, c2
                            else:
                                coloring[i1], coloring[i2] = c1, c2
            
            # If we've reached the theoretical minimum, stop
            if best_colors <= 13:
                break
    
    # Phase 3: Final high-precision polish - aggressive color elimination
    if best_colors > 13 and time.time() < deadline - 3:
        coloring = best_coloring.copy()
        current_colors = best_colors
        
        # Try to eliminate each color one at a time
        for _ in range(50000):
            if time.time() > deadline - 2:
                break
            
            unique, counts = np.unique(coloring, return_counts=True)
            if len(unique) == 0:
                break
            
            # Sort colors by frequency (try to eliminate most common first)
            order = np.argsort(-counts)
            
            improved_this_round = False
            for oi in order:
                target_color = int(unique[oi])
                target_edges = np.where(coloring == target_color)[0]
                
                if len(target_edges) == 0:
                    continue
                
                # Try to recolor all edges of target_color to other colors
                success = True
                old_colors = {}
                
                for ei in target_edges:
                    ei = int(ei)
                    old_colors[ei] = int(coloring[ei])
                    
                    # Find a valid new color
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
                    # Verify full validity
                    if check_valid(coloring):
                        new_count = count_colors(coloring)
                        if new_count < current_colors:
                            current_colors = new_count
                            best_colors = new_count
                            best_coloring = coloring.copy()
                            improved_this_round = True
                            break
            
            if not improved_this_round:
                # Try random single-edge recoloring as fallback
                target_color = int(unique[order[0]])
                target_edges = np.where(coloring == target_color)[0]
                if len(target_edges) > 0:
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
    # Ensure colors are in valid range and remap to minimize
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
