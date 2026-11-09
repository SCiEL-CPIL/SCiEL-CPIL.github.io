import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def run_vertex_sa(target_n, deadline, initial_colors=None):
        """
        Simulated Annealing with vertex-level moves and incremental violation tracking.
        A move flips all triples containing a randomly chosen vertex.
        Returns (colors, num_violations)
        """
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}
        
        # Build adjacency: for each vertex, list of triple indices containing it
        vertex_triples = [[] for _ in range(target_n)]
        for i, (a, b, c) in enumerate(triple_list):
            vertex_triples[a].append(i)
            vertex_triples[b].append(i)
            vertex_triples[c].append(i)
        
        # Color array: 0=red, 1=blue
        if initial_colors is None:
            color = [random.choice([0, 1]) for _ in range(num_triples)]
        else:
            color = list(initial_colors)
        
        quad_list = list(combinations(range(target_n), 4))
        quint_list = list(combinations(range(target_n), 5))
        
        # Precompute triple indices for each quad and quint
        quad_triples = []
        for quad in quad_list:
            a, b, c, d = quad
            quad_triples.append([
                triple_to_idx[(a,b,c)],
                triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)],
                triple_to_idx[(b,c,d)]
            ])
        
        quint_triples = []
        for quint in quint_list:
            quint_triples.append([triple_to_idx[t] for t in combinations(quint, 3)])
        
        # Precompute which quads/quints contain each vertex
        vertex_quads = [[] for _ in range(target_n)]
        for qi, quad in enumerate(quad_list):
            for v in quad:
                vertex_quads[v].append(qi)
        
        vertex_quints = [[] for _ in range(target_n)]
        for qi, quint in enumerate(quint_list):
            for v in quint:
                vertex_quints[v].append(qi)
        
        # Track red count per quad and blue count per quint
        quad_red_counts = np.zeros(len(quad_list), dtype=np.int32)
        quint_blue_counts = np.zeros(len(quint_list), dtype=np.int32)
        
        for qi, t_indices in enumerate(quad_triples):
            quad_red_counts[qi] = sum(1 for i in t_indices if color[i] == 0)
        
        for qi, t_indices in enumerate(quint_triples):
            quint_blue_counts[qi] = sum(1 for i in t_indices if color[i] == 1)
        
        red_k4_violations = int(np.sum(quad_red_counts == 4))
        blue_k5_violations = int(np.sum(quint_blue_counts == 10))
        
        current_cost = red_k4_violations + blue_k5_violations
        best_cost = current_cost
        best_color = list(color)
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        
        for it in range(1000000):
            if time.time() > deadline - 2:
                break
            
            # Choose a random vertex to flip
            v = random.randint(0, target_n - 1)
            affected_triples = vertex_triples[v]
            
            if not affected_triples:
                continue
            
            # Save old colors and compute delta
            old_colors = {}
            for t_idx in affected_triples:
                old_colors[t_idx] = color[t_idx]
            
            # Flip colors
            for t_idx in affected_triples:
                color[t_idx] = 1 - color[t_idx]
            
            # Update quad red counts for quads containing v
            delta_red_violations = 0
            for qi in vertex_quads[v]:
                t_indices = quad_triples[qi]
                old_red = quad_red_counts[qi]
                # Recompute red count
                new_red = 0
                for i in t_indices:
                    if color[i] == 0:
                        new_red += 1
                quad_red_counts[qi] = new_red
                if old_red == 4 and new_red != 4:
                    delta_red_violations -= 1
                elif old_red != 4 and new_red == 4:
                    delta_red_violations += 1
            
            # Update quint blue counts for quints containing v
            delta_blue_violations = 0
            for qi in vertex_quints[v]:
                t_indices = quint_triples[qi]
                old_blue = quint_blue_counts[qi]
                # Recompute blue count
                new_blue = 0
                for i in t_indices:
                    if color[i] == 1:
                        new_blue += 1
                quint_blue_counts[qi] = new_blue
                if old_blue == 10 and new_blue != 10:
                    delta_blue_violations -= 1
                elif old_blue != 10 and new_blue == 10:
                    delta_blue_violations += 1
            
            delta = delta_red_violations + delta_blue_violations
            new_cost = current_cost + delta
            
            # Accept or reject
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                red_k4_violations += delta_red_violations
                blue_k5_violations += delta_blue_violations
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_color = list(color)
                    if best_cost == 0:
                        return best_color, 0
            else:
                # Revert colors and counts
                for t_idx in affected_triples:
                    color[t_idx] = old_colors[t_idx]
                
                # Revert quad red counts
                for qi in vertex_quads[v]:
                    t_indices = quad_triples[qi]
                    quad_red_counts[qi] = sum(1 for i in t_indices if color[i] == 0)
                
                # Revert quint blue counts
                for qi in vertex_quints[v]:
                    t_indices = quint_triples[qi]
                    quint_blue_counts[qi] = sum(1 for i in t_indices if color[i] == 1)
            
            T *= cooling
            if T < T_min:
                T = T_min
                if it % 50000 == 0:
                    T = 5.0
        
        return best_color, best_cost

    # Strategy: Try n=34 with multiple restarts using different initializations
    targets = [34, 33, 32]
    time_allocations = [30, 20, 15]
    
    best_found = (9, mod_color(9, 3))
    
    for i, n in enumerate(targets):
        if time.time() > deadline - 15:
            break
            
        alloc = time_allocations[i] if i < len(time_allocations) else 10
        current_deadline = time.time() + alloc
        
        max_restarts = 20
        found_valid = False
        
        for r in range(max_restarts):
            if time.time() > current_deadline - 2:
                break
            
            # Different initialization strategies
            if r % 5 == 0:
                # Modular by 3
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r % 5 == 1:
                # Modular by 5
                init = [0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            elif r % 5 == 2:
                # Random with 40% red
                init = [0 if random.random() < 0.4 else 1 for _ in combinations(range(n), 3)]
            elif r % 5 == 3:
                # Random with 50% red
                init = [0 if random.random() < 0.5 else 1 for _ in combinations(range(n), 3)]
            else:
                # Modular by 7
                init = [0 if (a+b+c) % 7 == 0 else 1 for (a,b,c) in combinations(range(n), 3)]
            
            colors, cost = run_vertex_sa(n, current_deadline, init)
            
            if cost == 0:
                if is_valid(n, colors):
                    best_found = (n, colors)
                    found_valid = True
                    break
        
        if found_valid:
            break

    return {'n': best_found[0], 'colors': best_found[1]}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
    if res['n'] > 9:
        print("Validating...")
        print(is_valid(res['n'], res['colors']))
