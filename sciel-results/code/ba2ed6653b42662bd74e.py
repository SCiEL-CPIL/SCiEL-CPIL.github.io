import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return
    
    # Precompute structures for a given n
    def prepare_data(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Red K4s: all 4-subsets
        red_k4s = []
        for quad in combinations(range(n), 4):
            edges = []
            for tri in combinations(quad, 3):
                edges.append(triple_to_idx[tri])
            red_k4s.append(edges)
            
        # Blue K5s: all 5-subsets
        blue_k5s = []
        for quint in combinations(range(n), 5):
            edges = []
            for tri in combinations(quint, 3):
                edges.append(triple_to_idx[tri])
            blue_k5s.append(edges)
            
        # For each edge, which K4s and K5s does it belong to?
        edge_to_k4s = [[] for _ in range(num_triples)]
        edge_to_k5s = [[] for _ in range(num_triples)]
        
        for i, edges in enumerate(red_k4s):
            for e in edges:
                edge_to_k4s[e].append(i)
                
        for i, edges in enumerate(blue_k5s):
            for e in edges:
                edge_to_k5s[e].append(i)
        
        return triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s

    # Biased initial coloring
    def initial_coloring(n, p_red=0.6, seed_val=0):
        rng = random.Random(seed_val)
        num_triples = len(list(combinations(range(n), 3)))
        return [0 if rng.random() < p_red else 1 for _ in range(num_triples)]

    def local_search(n, colors, time_limit=5.0, p_red=0.6):
        triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s = prepare_data(n)
        num_triples = len(triples)
        
        colors = colors[:]
        
        red_counts = [0] * len(red_k4s)
        blue_counts = [0] * len(blue_k5s)
        
        for i, edges in enumerate(red_k4s):
            red_counts[i] = sum(1 for e in edges if colors[e] == 0)
        for i, edges in enumerate(blue_k5s):
            blue_counts[i] = sum(1 for e in edges if colors[e] == 1)
        
        red_violations = sum(1 for c in red_counts if c == 4)
        blue_violations = sum(1 for c in blue_counts if c == 10)
        total_violations = red_violations + blue_violations
        
        if total_violations == 0:
            return colors
        
        start = time.time()
        stuck = 0
        best_violations = total_violations
        best_colors = colors[:]
        
        while time.time() - start < time_limit:
            if total_violations == 0:
                return colors
            
            e = random.randint(0, num_triples - 1)
            current_color = colors[e]
            new_color = 1 - current_color
            
            delta_red = 0
            delta_blue = 0
            
            for k4_idx in edge_to_k4s[e]:
                cnt = red_counts[k4_idx]
                if current_color == 0:
                    if cnt == 4: delta_red -= 1
                else:
                    if cnt == 3: delta_red += 1
                    
            for k5_idx in edge_to_k5s[e]:
                cnt = blue_counts[k5_idx]
                if current_color == 1:
                    if cnt == 10: delta_blue -= 1
                else:
                    if cnt == 9: delta_blue += 1
                    
            total_delta = delta_red + delta_blue
            
            accepted = False
            if total_delta < 0:
                accepted = True
                stuck = 0
            elif total_delta == 0:
                if random.random() < 0.5:
                    accepted = True
                else:
                    stuck += 1
            else:
                if total_delta <= 2 and random.random() < 0.1:
                    accepted = True
                    stuck += 1
                else:
                    stuck += 1
                    
            if accepted:
                colors[e] = new_color
                for k4_idx in edge_to_k4s[e]:
                    if current_color == 0: red_counts[k4_idx] -= 1
                    else: red_counts[k4_idx] += 1
                for k5_idx in edge_to_k5s[e]:
                    if current_color == 1: blue_counts[k5_idx] -= 1
                    else: blue_counts[k5_idx] += 1
                red_violations += delta_red
                blue_violations += delta_blue
                total_violations += total_delta
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
            else:
                stuck += 1
                
            if stuck > 200:
                # Restart with biased coloring
                colors = initial_coloring(n, p_red, random.randint(0, 1000000))
                for i, edges in enumerate(red_k4s):
                    red_counts[i] = sum(1 for e in edges if colors[e] == 0)
                for i, edges in enumerate(blue_k5s):
                    blue_counts[i] = sum(1 for e in edges if colors[e] == 1)
                red_violations = sum(1 for c in red_counts if c == 4)
                blue_violations = sum(1 for c in blue_counts if c == 10)
                total_violations = red_violations + blue_violations
                stuck = 0
        
        return best_colors if best_violations < total_violations else colors

    # Try increasing n
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    for n in range(10, 35):
        if time.time() > deadline - 5:
            break
        
        # Allocate time for this n
        t_start = time.time()
        t_end = min(deadline - 2, t_start + 2 if n < 20 else t_start + 5)
        
        found = False
        while time.time() < t_end and not found:
            colors = initial_coloring(n, p_red=0.6, seed_val=random.randint(0, 1000000))
            
            # Quick check
            triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s = prepare_data(n)
            red_violations = 0
            blue_violations = 0
            for i, edges in enumerate(red_k4s):
                if all(colors[e] == 0 for e in edges):
                    red_violations += 1
            for i, edges in enumerate(blue_k5s):
                if all(colors[e] == 1 for e in edges):
                    blue_violations += 1
                    
            if red_violations == 0 and blue_violations == 0:
                best_n = n
                best_colors = colors
                found = True
                break
            
            # Local search
            refined = local_search(n, colors, time_limit=t_end - time.time())
            
            # Verify refined
            red_violations = 0
            blue_violations = 0
            for i, edges in enumerate(red_k4s):
                if all(refined[e] == 0 for e in edges):
                    red_violations += 1
            for i, edges in enumerate(blue_k5s):
                if all(refined[e] == 1 for e in edges):
                    blue_violations += 1
                    
            if red_violations == 0 and blue_violations == 0:
                best_n = n
                best_colors = refined
                found = True
                break
    
    return {'n': best_n, 'colors': best_colors}
