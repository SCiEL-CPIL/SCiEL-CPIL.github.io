import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 1
    best_colors = [0]  # Trivial valid coloring for n=1 (no triples)
    
    # We will try n from 34 down to 20
    for n in range(34, 19, -1):
        if time.time() > deadline:
            break
            
        # Precompute structures for this n
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        num_k4 = len(k4s)
        
        k5s = list(combinations(range(n), 5))
        num_k5 = len(k5s)
        
        # edge_k4s[i] = list of k4 indices containing triple i
        # edge_k5s[i] = list of k5 indices containing triple i
        edge_k4s = [[] for _ in range(num_triples)]
        edge_k5s = [[] for _ in range(num_triples)]
        
        # k4_edges[i] = list of triple indices in k4 i
        k4_edges = [[] for _ in range(num_k4)]
        # k5_edges[i] = list of triple indices in k5 i
        k5_edges = [[] for _ in range(num_k5)]
        
        # Fill k4_edges and edge_k4s
        for k4_idx, k4 in enumerate(k4s):
            a, b, c, d = k4
            # 4 triples
            t1 = triple_index[(a, b, c)]
            t2 = triple_index[(a, b, d)]
            t3 = triple_index[(a, c, d)]
            t4 = triple_index[(b, c, d)]
            k4_edges[k4_idx] = [t1, t2, t3, t4]
            edge_k4s[t1].append(k4_idx)
            edge_k4s[t2].append(k4_idx)
            edge_k4s[t3].append(k4_idx)
            edge_k4s[t4].append(k4_idx)
            
        # Fill k5_edges and edge_k5s
        for k5_idx, k5 in enumerate(k5s):
            a, b, c, d, e = k5
            # 10 triples
            # Generate all combinations of 3 from the 5 vertices
            k5_triples = list(combinations(k5, 3))
            indices = []
            for t in k5_triples:
                t_idx = triple_index[t]
                indices.append(t_idx)
                edge_k5s[t_idx].append(k5_idx)
            k5_edges[k5_idx] = indices
            
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        red_counts = [0] * num_k4
        blue_counts = [0] * num_k5
        total_violations = 0
        
        # Compute initial counts
        for k4_idx in range(num_k4):
            count = 0
            for t_idx in k4_edges[k4_idx]:
                if colors[t_idx] == 0:
                    count += 1
            red_counts[k4_idx] = count
            if count == 4:
                total_violations += 1
                
        for k5_idx in range(num_k5):
            count = 0
            for t_idx in k5_edges[k5_idx]:
                if colors[t_idx] == 1:
                    count += 1
            blue_counts[k5_idx] = count
            if count == 10:
                total_violations += 1
                
        # Local Search with Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        stuck_counter = 0
        max_stuck = 2000
        
        while time.time() < deadline:
            if total_violations == 0:
                best_n = n
                best_colors = colors[:]
                break
                
            t = random.randint(0, num_triples - 1)
            current_color = colors[t]
            new_color = 1 - current_color
            
            delta = 0
            
            # Check K4s
            for k4_idx in edge_k4s[t]:
                if current_color == 0: # Was red, becoming blue
                    # red_count decreases by 1
                    if red_counts[k4_idx] == 4:
                        delta -= 1
                else: # Was blue, becoming red
                    # red_count increases by 1
                    if red_counts[k4_idx] == 3:
                        delta += 1
                    
            # Check K5s
            for k5_idx in edge_k5s[t]:
                if current_color == 1: # Was blue, becoming red
                    # blue_count decreases by 1
                    if blue_counts[k5_idx] == 10:
                        delta -= 1
                else: # Was red, becoming blue
                    # blue_count increases by 1
                    if blue_counts[k5_idx] == 9:
                        delta += 1
            
            # Decide acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                # Simulated annealing acceptance
                if T > T_min:
                    if random.random() < pow(2.71828, -delta / T):
                        accept = True
            
            if accept:
                # Update colors
                colors[t] = new_color
                
                # Update counts
                for k4_idx in edge_k4s[t]:
                    if current_color == 0:
                        red_counts[k4_idx] -= 1
                    else:
                        red_counts[k4_idx] += 1
                        
                for k5_idx in edge_k5s[t]:
                    if current_color == 1:
                        blue_counts[k5_idx] -= 1
                    else:
                        blue_counts[k5_idx] += 1
                        
                total_violations += delta
                stuck_counter = 0
            else:
                stuck_counter += 1
                
            T *= cooling
            if T < T_min:
                T = T_min
                
            if stuck_counter > max_stuck:
                # If stuck, break to try next n or restart?
                # Breaking is safer to avoid wasting time on a dead end
                break
                
        if total_violations == 0:
            break # Found valid coloring for this n
            
    return {'n': best_n, 'colors': best_colors}
