import time
import random
import math
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    n = 16
    m = 120  # number of edges
    
    # Precompute all K4s and their edge indices
    k4s = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            idx += 1
    
    for a, b, c, d in combinations(range(n), 4):
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)
    
    num_k4s = len(k4s)  # C(16,4) = 1820
    
    # Precompute for each edge, which K4s it belongs to and its position in that K4
    edge_to_k4s = [[] for _ in range(m)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for pos, ei in enumerate(edge_idxs):
            edge_to_k4s[ei].append((k4_idx, pos))
    
    # Precompute edge-to-vertex mapping
    edge_to_vertices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_vertices.append((i, j))
            idx += 1
    
    # Precompute: for each edge, the set of other vertices (n-2 of them)
    # An edge (i,j) is in a K4 with any two other vertices
    edge_to_other_vertices = []
    for i, j in edge_to_vertices:
        others = [v for v in range(n) if v != i and v != j]
        edge_to_other_vertices.append(others)
    
    # Generate diverse initial candidates
    def gen_modular(mod, offset=0):
        return [((i + j) + offset) % mod for i in range(n) for j in range(i + 1, n)]
    
    def gen_random():
        return [random.randint(0, 12) for _ in range(m)]
    
    def gen_greedy():
        """Greedy: assign colors to try to minimize distinct while keeping valid."""
        # Start with (i+j)%13 and try to merge
        c = gen_modular(13)
        return c
    
    candidates = []
    for mod in [13, 14, 15]:
        candidates.append(gen_modular(mod))
    for offset in [1, 2, 3]:
        candidates.append(gen_modular(13, offset))
        candidates.append(gen_modular(14, offset))
    
    random.seed(42)
    for _ in range(8):
        candidates.append(gen_random())
    
    # Evaluate a coloring
    def evaluate(colors):
        violations = 0
        for edge_idxs in k4s:
            if len(set(colors[ei] for ei in edge_idxs)) < 5:
                violations += 1
        distinct = len(set(colors))
        return violations, distinct
    
    # Find best initial candidate
    best_colors = None
    best_violations = float('inf')
    best_distinct = float('inf')
    
    for c in candidates:
        v, d = evaluate(c)
        if v < best_violations or (v == best_violations and d < best_distinct):
            best_violations = v
            best_distinct = d
            best_colors = c[:]
    
    # If no valid solution found, use modular 13 as fallback
    if best_violations > 0:
        # Try harder with more candidates
        for _ in range(20):
            c = gen_random()
            v, d = evaluate(c)
            if v < best_violations or (v == best_violations and d < best_distinct):
                best_violations = v
                best_distinct = d
                best_colors = c[:]
        
        if best_violations > 0:
            # Use (i+j)%13 as guaranteed valid-ish fallback
            best_colors = gen_modular(13)
            best_violations, best_distinct = evaluate(best_colors)
    
    colors = best_colors[:]
    current_violations = best_violations
    current_distinct = best_distinct
    
    # Data structures for incremental updates
    # For each K4, track the multiset of colors (as a list of 6)
    k4_edge_colors = [[colors[ei] for ei in edge_idxs] for edge_idxs in k4s]
    
    # Color counts
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    def k4_is_violation(k4_colors):
        return len(set(k4_colors)) < 5
    
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
            old_violation = k4_is_violation(old_k4_colors)
            
            old_k4_colors[pos] = new_color
            new_violation = k4_is_violation(old_k4_colors)
            
            if old_violation and not new_violation:
                current_violations -= 1
            elif not old_violation and new_violation:
                current_violations += 1
            
        colors[edge_idx] = new_color
    
    def revert_state(edge_idx, old_color, new_color):
        update_state(edge_idx, new_color, old_color)
    
    random.seed(123)
    
    # Phase 1: Simulated Annealing to reach 0 violations
    if current_violations > 0:
        temperature = 20.0
        cooling_rate = 0.999
        min_temperature = 0.1
        max_steps = 500000
        
        step = 0
        while time.time() < deadline - 30 and temperature > min_temperature and step < max_steps:
            edge_idx = random.randint(0, m - 1)
            old_color = colors[edge_idx]
            
            # Sample new color
            if random.random() < 0.7 and color_counts:
                new_color = random.choice(list(color_counts.keys()))
            else:
                new_color = random.randint(0, 14)
                
            if new_color == old_color:
                continue
                
            old_cost = current_violations * 100 + current_distinct
            
            update_state(edge_idx, old_color, new_color)
            
            new_cost = current_violations * 100 + current_distinct
            delta_cost = new_cost - old_cost
            
            if delta_cost < 0:
                pass
            else:
                prob = math.exp(-delta_cost / temperature)
                if random.random() >= prob:
                    revert_state(edge_idx, old_color, new_color)
            
            temperature *= cooling_rate
            step += 1
            
            if current_violations == 0:
                break
    
    # Phase 2: Local search to minimize distinct colors (starting from valid state)
    if current_violations == 0:
        best_distinct = current_distinct
        best_colors = colors[:]
        
        # Greedy color elimination
        max_rounds = 100
        for round_num in range(max_rounds):
            if time.time() > deadline - 15:
                break
            if current_distinct <= 13:
                break
                
            used_colors = sorted(color_counts.keys())
            if len(used_colors) <= best_distinct:
                break
            
            # Pick the color with fewest edges to try to eliminate
            target_color = min(used_colors, key=lambda c: color_counts[c])
            edges_with_color = [i for i, c in enumerate(colors) if c == target_color]
            
            improved = False
            for edge_idx in edges_with_color:
                if time.time() > deadline - 15:
                    break
                if target_color not in color_counts:
                    break
                    
                old_color = colors[edge_idx]
                # Try each other color
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
                # Try random perturbation to escape local optimum
                if random.random() < 0.5 and time.time() < deadline - 20:
                    for _ in range(5):
                        e = random.randint(0, m - 1)
                        old_c = colors[e]
                        new_c = random.randint(0, 14)
                        if new_c != old_c:
                            update_state(e, old_c, new_c)
                            if current_violations == 0 and current_distinct < best_distinct:
                                best_distinct = current_distinct
                                best_colors = colors[:]
                            # Keep the perturbation if it didn't increase violations too much
                            if current_violations > 5:
                                revert_state(e, new_c, old_c)
                    # Re-evaluate
                    if current_violations > 0:
                        # Try to fix
                        for _ in range(1000):
                            if time.time() > deadline - 15:
                                break
                            if current_violations == 0:
                                break
                            e = random.randint(0, m - 1)
                            old_c = colors[e]
                            new_c = random.choice(list(color_counts.keys())) if color_counts else 0
                            if new_c != old_c:
                                update_state(e, old_c, new_c)
                                if current_violations > 0:
                                    revert_state(e, new_c, old_c)
                else:
                    break
            
            if current_violations == 0 and current_distinct < best_distinct:
                best_distinct = current_distinct
                best_colors = colors[:]
        
        # Phase 3: Fine-grained SA on valid solutions to reduce colors
        if current_violations == 0:
            temperature = 5.0
            cooling_rate = 0.9995
            min_temp = 0.01
            step = 0
            while time.time() < deadline - 5 and temperature > min_temp and step < 300000:
                edge_idx = random.randint(0, m - 1)
                old_color = colors[edge_idx]
                
                if random.random() < 0.8 and color_counts:
                    new_color = random.choice(list(color_counts.keys()))
                else:
                    new_color = random.randint(0, 14)
                    
                if new_color == old_color:
                    continue
                    
                old_cost = current_violations * 1000 + current_distinct
                update_state(edge_idx, old_color, new_color)
                new_cost = current_violations * 1000 + current_distinct
                delta_cost = new_cost - old_cost
                
                if delta_cost < 0:
                    if current_violations == 0 and current_distinct < best_distinct:
                        best_distinct = current_distinct
                        best_colors = colors[:]
                else:
                    prob = math.exp(-delta_cost / temperature)
                    if random.random() >= prob:
                        revert_state(edge_idx, old_color, new_color)
                
                temperature *= cooling_rate
                step += 1
    
    # Final verification
    if best_colors is None:
        best_colors = gen_modular(13)
    
    v, d = evaluate(best_colors)
    if v == 0:
        return best_colors
    
    # Fallback: find any valid candidate
    for c in candidates:
        v_c, d_c = evaluate(c)
        if v_c == 0:
            return c
    
    # Last resort: (i+j)%13
    fallback = gen_modular(13)
    v_f, d_f = evaluate(fallback)
    if v_f == 0:
        return fallback
    
    # Absolute last resort
    return gen_modular(14)
