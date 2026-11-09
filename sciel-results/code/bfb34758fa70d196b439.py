import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 14
    m = 35
    max_val = 10000
    
    candidates = [
        list(range(1, 15)),
        [2 * i for i in range(1, 15)],
        [3 * i for i in range(1, 15)],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        sorted(random.sample(range(1, 30), 14)),
        sorted(random.sample(range(1, 40), 14)),
    ]
    
    seen = set()
    unique_candidates = []
    for v in candidates:
        key = tuple(sorted(v))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
            
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    all_edges = list(combinations(range(n), 2))
    
    for vals in unique_candidates:
        if time.time() > deadline:
            break
            
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
            
        edge_data = {}
        for u, v in all_edges:
            edge_data[(u, v)] = (vals[u] + vals[v], vals[u] * vals[v])
            
        # Greedy Init
        selected = set()
        sums = set()
        prods = set()
        remaining = list(all_edges)
        
        sum_counts = {}
        prod_counts = {}
        for e in all_edges:
            s, p = edge_data[e]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        remaining.sort(key=lambda e: (sum_counts[edge_data[e][0]], prod_counts[edge_data[e][1]]), reverse=True)
        
        while len(selected) < m:
            best_e = remaining[0]
            best_sc = float('inf')
            for e in remaining[:20]:
                s, p = edge_data[e]
                ns = len(sums | {s})
                np = len(prods | {p})
                sc = max(ns, np)
                if sc < best_sc:
                    best_sc = sc
                    best_e = e
            selected.add(best_e)
            remaining.remove(best_e)
            s, p = edge_data[best_e]
            sums.add(s)
            prods.add(p)
            
        # Local Search
        s_counts = {}
        p_counts = {}
        for e in selected:
            s, p = edge_data[e]
            s_counts[s] = s_counts.get(s, 0) + 1
            p_counts[p] = p_counts.get(p, 0) + 1
            
        cur_score = max(len(s_counts), len(p_counts))
        
        # Simulated Annealing / Random Restart Local Search
        T = 1.0
        for it in range(20000):
            if time.time() > deadline:
                break
                
            e_out = random.choice(list(selected))
            e_in = random.choice([e for e in all_edges if e not in selected])
            
            s_out, p_out = edge_data[e_out]
            s_in, p_in = edge_data[e_in]
            
            # Calculate new distinct counts
            new_s_counts = dict(s_counts)
            new_p_counts = dict(p_counts)
            
            new_s_counts[s_out] -= 1
            if new_s_counts[s_out] == 0:
                del new_s_counts[s_out]
            new_p_counts[p_out] -= 1
            if new_p_counts[p_out] == 0:
                del new_p_counts[p_out]
            
            if s_in not in new_s_counts:
                new_s_counts[s_in] = 1
            else:
                new_s_counts[s_in] += 1
                
            if p_in not in new_p_counts:
                new_p_counts[p_in] = 1
            else:
                new_p_counts[p_in] += 1
                
            new_score = max(len(new_s_counts), len(new_p_counts))
            
            delta = new_score - cur_score
            
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
                # Accept
                selected.discard(e_out)
                selected.add(e_in)
                
                s_counts = new_s_counts
                p_counts = new_p_counts
                cur_score = new_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_values = vals
                    best_edges = list(selected)
                    
            # Cool down
            T *= 0.999
            if T < 0.01:
                T = 0.01

    if best_values is None:
        best_values = unique_candidates[0]
        best_edges = all_edges[:m]
        
    values = [int(v) for v in best_values]
    edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': values, 'edges': edges}
