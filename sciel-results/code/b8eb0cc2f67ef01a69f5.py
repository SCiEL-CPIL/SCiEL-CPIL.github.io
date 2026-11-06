import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    # Precompute all K4s and their edge indices
    k4s = []
    edge_index = {}
    edge_ij = []  # map edge index -> (i, j)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edge_ij.append((i, j))
            idx += 1
    
    for a,b,c,d in combinations(range(n), 4):
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)
    
    # Precompute for each edge, which K4s it belongs to and its position in that K4
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for pos, ei in enumerate(edge_idxs):
            edge_to_k4s[ei].append((k4_idx, pos))
            
    # Initial candidates
    candidates = []
    for mod in [13, 14, 15, 17, 19, 23]:
        c = [(i + j) % mod for i in range(n) for j in range(i + 1, n)]
        candidates.append(c)
    
    random.seed(42)
    # Add a few random perturbations of the best modular
    # First find best modular
    def evaluate(colors):
        violations = 0
        for edge_idxs in k4s:
            if len(set(colors[ei] for ei in edge_idxs)) < 5:
                violations += 1
        distinct = len(set(colors))
        return violations, distinct
        
    best_mod_color = None
    best_mod_v = float('inf')
    best_mod_d = float('inf')
    for c in candidates:
        v, d = evaluate(c)
        if v < best_mod_v or (v == best_mod_v and d < best_mod_d):
            best_mod_v = v
            best_mod_d = d
            best_mod_color = c
            
    # Create random perturbations of best_mod_color
    for _ in range(5):
        c = best_mod_color[:]
        for _ in range(10):
            e = random.randint(0, 119)
            c[e] = random.randint(0, 15)
        candidates.append(c)

    # Pick the absolute best candidate to start SA
    best_colors = None
    best_violations = float('inf')
    best_distinct = float('inf')
    
    for c in candidates:
        v, d = evaluate(c)
        score = (v, d)
        if best_colors is None or score < (best_violations, best_distinct):
            best_colors = c
            best_violations = v
            best_distinct = d
            
    colors = best_colors[:]
    current_violations = best_violations
    current_distinct = best_distinct
    
    # Data structures for incremental updates
    k4_edge_colors = [ [colors[ei] for ei in edge_idxs] for edge_idxs in k4s ]
    
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
        
    def update_state(edge_idx, old_color, new_color):
        nonlocal current_violations, current_distinct
        
        # Update color counts
        color_counts[old_color] -= 1
        if color_counts[old_color] == 0:
            del color_counts[old_color]
            current_distinct -= 1
            
        color_counts[new_color] = color_counts.get(new_color, 0) + 1
        if color_counts[new_color] == 1:
            current_distinct += 1
            
        # Update K4s
        for k4_idx, pos in edge_to_k4s[edge_idx]:
            old_k4_colors = k4_edge_colors[k4_idx]
            old_set = set(old_k4_colors)
            old_violation = 1 if len(old_set) < 5 else 0
            
            old_k4_colors[pos] = new_color
            new_set = set(old_k4_colors)
            new_violation = 1 if len(new_set) < 5 else 0
            
            if old_violation and not new_violation:
                current_violations -= 1
            elif not old_violation and new_violation:
                current_violations += 1
            
        colors[edge_idx] = new_color

    def revert_state(edge_idx, old_color, new_color):
        update_state(edge_idx, new_color, old_color)
        
    random.seed(42)
    
    # Simulated Annealing
    temperature = 50.0
    cooling_rate = 0.995
    min_temperature = 0.5
    
    while time.time() < deadline and temperature > min_temperature:
        edge_idx = random.randint(0, 119)
        old_color = colors[edge_idx]
        
        # Sample new color
        # 80% from existing, 20% from 0..15
        if random.random() < 0.8 and color_counts:
            new_color = random.choice(list(color_counts.keys()))
        else:
            new_color = random.randint(0, 15)
            
        if new_color == old_color:
            continue
            
        old_cost = current_violations * 1000 + current_distinct
        
        update_state(edge_idx, old_color, new_color)
        
        new_cost = current_violations * 1000 + current_distinct
        
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0:
            pass
        else:
            if temperature > 0:
                prob = np.exp(-delta_cost / temperature)
                if random.random() < prob:
                    pass
                else:
                    revert_state(edge_idx, old_color, new_color)
            
        temperature *= cooling_rate
        
        if current_violations == 0 and current_distinct < best_distinct:
            best_distinct = current_distinct
            best_colors = colors[:]
            
    # Final polish: Hill climbing on valid solutions
    if current_violations == 0:
        # Try to reduce distinct colors
        for _ in range(1000):
            if time.time() > deadline:
                break
            if not color_counts:
                break
            used_colors = list(color_counts.keys())
            if len(used_colors) <= best_distinct:
                break
            target_color = min(used_colors, key=lambda c: color_counts[c])
            edges_with_color = [i for i, c in enumerate(colors) if c == target_color]
            
            improved = False
            for edge_idx in edges_with_color:
                old_color = colors[edge_idx]
                for new_color in used_colors:
                    if new_color == target_color:
                        continue
                    
                    old_cost = current_violations * 1000 + current_distinct
                    update_state(edge_idx, old_color, new_color)
                    new_cost = current_violations * 1000 + current_distinct
                    
                    if new_cost < old_cost:
                        improved = True
                        break
                    else:
                        revert_state(edge_idx, old_color, new_color)
                
                if improved:
                    break
            
            if not improved:
                break
            if current_violations == 0 and current_distinct < best_distinct:
                best_distinct = current_distinct
                best_colors = colors[:]
    else:
        # If current is invalid, check if we ever found a valid one
        pass

    # Verify and return
    if best_colors is None:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    v, d = evaluate(best_colors)
    if v == 0:
        return best_colors
    else:
        # Fallback to best valid candidate
        for c in candidates:
            v_c, d_c = evaluate(c)
            if v_c == 0:
                return c
        return candidates[0]
