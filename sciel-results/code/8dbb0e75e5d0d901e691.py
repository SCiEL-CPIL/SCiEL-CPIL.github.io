import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute K4s and their edge indices
    edge_index = {}
    idx = 0
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            edges.append((i, j))
            idx += 1
            
    k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    # Map edge index to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(120)]
    
    for k_idx, quad in enumerate(k4s):
        a, b, c, d = quad
        e = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4_edges.append(e)
        for e_idx in e:
            edge_to_k4s[e_idx].append(k_idx)
            
    # Baseline solution: 17 colors
    baseline = [(i + j) % 17 for i, j in edges]
    best_sol = baseline[:]
    best_val = 17
    
    def init_state(colors):
        """
        Initialize k4_distinct_counts and cur_viol for a given color array.
        """
        k4_distinct_counts = [0] * len(k4s)
        cur_viol = 0
        for k_idx, e in enumerate(k4_edges):
            c0 = colors[e[0]]
            c1 = colors[e[1]]
            c2 = colors[e[2]]
            c3 = colors[e[3]]
            c4 = colors[e[4]]
            c5 = colors[e[5]]
            distinct = len({c0, c1, c2, c3, c4, c5})
            k4_distinct_counts[k_idx] = distinct
            if distinct < 5:
                cur_viol += 1
        return k4_distinct_counts, cur_viol

    def update_violation(e_idx, old_color, new_color, colors, k4_distinct_counts, cur_viol):
        """
        Update state after changing edge e_idx from old_color to new_color.
        Returns new cur_viol.
        """
        if old_color == new_color:
            return cur_viol
            
        # Update colors array
        colors[e_idx] = new_color
        
        for k_idx in edge_to_k4s[e_idx]:
            e = k4_edges[k_idx]
            c0 = colors[e[0]]
            c1 = colors[e[1]]
            c2 = colors[e[2]]
            c3 = colors[e[3]]
            c4 = colors[e[4]]
            c5 = colors[e[5]]
            
            old_distinct = k4_distinct_counts[k_idx]
            new_distinct = len({c0, c1, c2, c3, c4, c5})
            
            if old_distinct < 5 and new_distinct >= 5:
                cur_viol -= 1
            elif old_distinct >= 5 and new_distinct < 5:
                cur_viol += 1
                
            k4_distinct_counts[k_idx] = new_distinct
            
        return cur_viol

    # Search for K=16, 15, 14
    for K in [16, 15, 14]:
        if time.time() > deadline - 30:
            break
            
        found = False
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
                
            cur = baseline[:]
            
            # Map colors >= K to random colors in 0..K-1
            for i in range(120):
                if cur[i] >= K:
                    cur[i] = random.randint(0, K - 1)
                    
            k4_distinct_counts, cur_viol = init_state(cur)
            
            if cur_viol == 0:
                best_sol = cur[:]
                best_val = len(set(cur))
                found = True
                break
                
            # Local search
            max_iters = 5000
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                e_idx = random.randint(0, 119)
                old_color = cur[e_idx]
                new_color = random.randint(0, K - 1)
                
                if new_color == old_color:
                    continue
                    
                # Tentative update
                # We need to revert if worse
                old_viol = cur_viol
                cur_viol = update_violation(e_idx, old_color, new_color, cur, k4_distinct_counts, cur_viol)
                
                if cur_viol <= old_viol:
                    if cur_viol == 0:
                        best_sol = cur[:]
                        best_val = len(set(cur))
                        found = True
                        break
                else:
                    # Revert
                    cur_viol = update_violation(e_idx, new_color, old_color, cur, k4_distinct_counts, cur_viol)
                    
            if found:
                break
                
    if len(set(best_sol)) < best_val:
        best_val = len(set(best_sol))
        
    return best_sol
