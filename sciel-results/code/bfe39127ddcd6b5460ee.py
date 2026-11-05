import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # List of all edges (i, j)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # List of all K4s (sets of 4 vertices)
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)  # 1820
    
    # For each K4, find the indices of its 6 edges
    k4_edge_indices = []
    for k4 in k4s:
        vs = list(k4)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(vs[i], vs[j])])
        k4_edge_indices.append(e_indices)
        
    # For each edge, find which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)

    def structured_init(max_colors):
        """
        Use vertex labels to derive edge colors via a nonlinear function.
        Each vertex gets a label from Z_p, and edge color = f(label_i, label_j).
        Try several structured approaches.
        """
        # Approach: assign each vertex a distinct value in Z_p for small prime p
        # Edge color = (a * u * v + b * (u + v) + c) mod max_colors
        # This creates structure that may help K4s have diverse colors
        
        best_init = None
        best_deficit = float('inf')
        
        for trial in range(50):
            # Random vertex labels
            labels = list(range(n))
            random.shuffle(labels)
            # Random coefficients
            a = random.randint(1, max_colors - 1)
            b = random.randint(0, max_colors - 1)
            c = random.randint(0, max_colors - 1)
            
            colors = [0] * num_edges
            for idx, (i, j) in enumerate(edges):
                u, v = labels[i], labels[j]
                colors[idx] = (a * u * v + b * (u + v) + c) % max_colors
            
            # Compute deficit
            deficit = 0
            for e_indices in k4_edge_indices:
                distinct = len(set(colors[idx] for idx in e_indices))
                if distinct < 5:
                    deficit += 5 - distinct
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_init = colors[:]
            if deficit == 0:
                break
        
        # If still high deficit, add random perturbation
        if best_deficit > 0:
            for _ in range(200):
                e_idx = random.randint(0, num_edges - 1)
                new_c = random.randint(0, max_colors - 1)
                best_init[e_idx] = new_c
            
            # Quick greedy repair
            k4_order = list(range(num_k4s))
            random.shuffle(k4_order)
            for k4_idx in k4_order:
                e_indices = k4_edge_indices[k4_idx]
                cur_colors = [best_init[idx] for idx in e_indices]
                distinct = set(cur_colors)
                if len(distinct) >= 5:
                    continue
                for pos, e_idx in enumerate(e_indices):
                    old_c = best_init[e_idx]
                    for new_c in range(max_colors):
                        if new_c == old_c or new_c in distinct:
                            continue
                        best_init[e_idx] = new_c
                        distinct.add(new_c)
                        break
                    if len(distinct) >= 5:
                        break
        
        return best_init

    def local_search_smooth(max_colors, time_limit, init_colors=None):
        """
        Local search minimizing total_deficit = sum(max(0, 5 - distinct_colors) for all K4s)
        """
        if init_colors is not None:
            colors = init_colors[:]
        else:
            colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Precompute current state
        k4_counts = []
        k4_distinct = []
        k4_deficit = []
        total_deficit = 0
        
        for e_indices in k4_edge_indices:
            counts = np.zeros(max_colors, dtype=np.int8)
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = int(np.count_nonzero(counts))
            deficit = 5 - distinct if distinct < 5 else 0
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            k4_deficit.append(deficit)
            total_deficit += deficit
            
        best_deficit = total_deficit
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 3.0
        min_temp = 0.01
        cool_rate = 0.9997
        
        # Track stagnation
        stagnation_count = 0
        max_stagnation = 3000
        
        # Precompute edge weights for weighted selection
        # Each edge is in C(14,2) = 91 K4s
        edge_k4_count = [len(edge_to_k4s[e]) for e in range(num_edges)]
        
        # Main Loop
        iterations = 0
        while time.time() < start_time + time_limit:
            iterations += 1
            if total_deficit == 0:
                return colors, 0
            
            # Pick an edge: biased toward edges in K4s with high deficit
            if total_deficit > 0 and iterations % 10 == 0:
                # Recompute edge weights every 10 iterations
                edge_weights = np.zeros(num_edges, dtype=np.float32)
                for k4_idx in range(num_k4s):
                    d = k4_deficit[k4_idx]
                    if d > 0:
                        for e_idx in k4_edge_indices[k4_idx]:
                            edge_weights[e_idx] += d
                total_w = edge_weights.sum()
                if total_w > 0:
                    probs = edge_weights / total_w
                    probs += 1e-7
                    probs /= probs.sum()
                    e_idx = int(np.random.choice(num_edges, p=probs))
                else:
                    e_idx = random.randint(0, num_edges - 1)
            else:
                e_idx = random.randint(0, num_edges - 1)
            
            old_color = colors[e_idx]
            
            # Pick a new color
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue
            
            # Calculate Delta Deficit
            delta_deficit = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Store old states for revert if needed
            old_states = []
            
            for k4_idx in affected_k4s:
                counts = k4_counts[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_def = k4_deficit[k4_idx]
                old_states.append((k4_idx, old_d, old_def))
                
                # Update counts
                counts[old_color] -= 1
                counts[new_color] += 1
                
                # Update distinct
                new_d = old_d
                if counts[old_color] == 0:
                    new_d -= 1
                if counts[new_color] == 1:  # It was 0 before
                    new_d += 1
                
                # Update deficit
                new_def = 5 - new_d if new_d < 5 else 0
                
                k4_distinct[k4_idx] = new_d
                k4_deficit[k4_idx] = new_def
                
                delta_deficit += new_def - old_def
            
            # Acceptance Criteria
            accept = False
            if delta_deficit <= 0:
                accept = True
                if delta_deficit < 0:
                    stagnation_count = 0
            else:
                # Avoid overflow in exp
                if delta_deficit < 100 * temp:
                    if random.random() < math.exp(-delta_deficit / temp):
                        accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_deficit += delta_deficit
                if total_deficit < best_deficit:
                    best_deficit = total_deficit
                    best_colors = colors[:]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                # Revert counts
                for k4_idx, old_d, old_def in old_states:
                    counts = k4_counts[k4_idx]
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
                    k4_deficit[k4_idx] = old_def
                stagnation_count += 1
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp
            
            # Restart if stuck
            if stagnation_count > max_stagnation:
                # Perturb: change a random subset of edges
                num_perturb = random.randint(8, 20)
                for _ in range(num_perturb):
                    e_idx = random.randint(0, num_edges - 1)
                    new_c = random.randint(0, max_colors - 1)
                    old_c = colors[e_idx]
                    if new_c == old_c:
                        continue
                    # Update state
                    for k4_idx in edge_to_k4s[e_idx]:
                        counts = k4_counts[k4_idx]
                        old_d = k4_distinct[k4_idx]
                        old_def = k4_deficit[k4_idx]
                        counts[old_c] -= 1
                        counts[new_c] += 1
                        new_d = old_d
                        if counts[old_c] == 0:
                            new_d -= 1
                        if counts[new_c] == 1:
                            new_d += 1
                        new_def = 5 - new_d if new_d < 5 else 0
                        k4_distinct[k4_idx] = new_d
                        k4_deficit[k4_idx] = new_def
                        total_deficit += new_def - old_def
                    colors[e_idx] = new_c
                temp = max(2.0, temp * 5)
                stagnation_count = 0
                
        return best_colors, best_deficit

    # Try C=13 with multiple restarts using structured initialization
    found = False
    for attempt in range(5):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        t_alloc = min(20.0, remaining - 5.0)
        if t_alloc < 3:
            break
        
        # Use structured init
        init = structured_init(13)
        
        sol, deficit = local_search_smooth(13, t_alloc, init)
        if deficit == 0:
            return sol
    
    # Try C=14 for remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        for attempt in range(3):
            t_alloc = min(20.0, remaining - 5.0)
            if t_alloc < 3:
                break
            init = structured_init(14)
            sol, deficit = local_search_smooth(14, t_alloc, init)
            if deficit == 0:
                return sol
    
    # Try C=15 for remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        init = structured_init(15)
        sol, deficit = local_search_smooth(15, remaining - 2.0, init)
        if deficit == 0:
            return sol

    # Fallback: Return a valid solution with more colors
    # The seed (i+j)%17 is valid. 
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    return seed_colors
