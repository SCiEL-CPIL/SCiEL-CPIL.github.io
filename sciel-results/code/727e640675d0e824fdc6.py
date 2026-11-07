import random
import math

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    adj_sets = [set() for _ in range(n)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    def evaluate(subset_list):
        s = set(subset_list)
        degrees = []
        for v in s:
            deg = len(adj_sets[v] & s)
            degrees.append(deg)
        return len(set(degrees))
    
    def local_search(subset):
        current_subset = set(subset)
        current_score = evaluate(list(current_subset))
        improved = True
        while improved:
            improved = False
            in_list = list(current_subset)
            out_list = [v for v in range(n) if v not in current_subset]
            best_swap = None
            best_swap_score = current_score
            for v_out in in_list:
                for v_in in out_list:
                    new_subset = set(current_subset)
                    new_subset.remove(v_out)
                    new_subset.add(v_in)
                    score = evaluate(list(new_subset))
                    if score > best_swap_score:
                        best_swap_score = score
                        best_swap = (v_out, v_in, new_subset)
            if best_swap:
                v_out, v_in, new_subset = best_swap
                current_subset = new_subset
                current_score = best_swap_score
                improved = True
        if len(current_subset) < n:
            out_list = [v for v in range(n) if v not in current_subset]
            for v_in in out_list:
                new_subset = set(current_subset)
                new_subset.add(v_in)
                score = evaluate(list(new_subset))
                if score > current_score:
                    current_subset = new_subset
                    current_score = score
                    improved = True
        if len(current_subset) > min_vertices:
            in_list = list(current_subset)
            for v_out in in_list:
                new_subset = set(current_subset)
                new_subset.remove(v_out)
                score = evaluate(list(new_subset))
                if score > current_score:
                    current_subset = new_subset
                    current_score = score
                    improved = True
        return list(current_subset), current_score
    
    best_subset = None
    best_score = -1
    
    for restart in range(50):
        strategy = restart % 5
        if strategy == 0:
            size = min_vertices
            current_subset = set(random.sample(range(n), size))
        elif strategy == 1:
            size = random.randint(min_vertices, n)
            current_subset = set(random.sample(range(n), size))
        elif strategy == 2:
            current_subset = set(range(n))
        elif strategy == 3:
            degs = [(len(adj_sets[v]), v) for v in range(n)]
            degs.sort(reverse=True)
            size = random.randint(min_vertices, min(n, 12))
            current_subset = set([v for _, v in degs[:size]])
        else:
            size = random.randint(min_vertices, n)
            current_subset = set(random.sample(range(n), size))
        
        current_subset, current_score = local_search(current_subset)
        
        if current_score > best_score:
            best_score = current_score
            best_subset = list(current_subset)
        
        T = 3.0
        cooling = 0.999
        min_T = 0.001
        max_iters = 20000
        
        for i in range(max_iters):
            in_list = list(current_subset)
            out_list = [v for v in range(n) if v not in current_subset]
            
            if not out_list:
                if len(current_subset) <= min_vertices:
                    break
                swap_out = random.choice(in_list)
                new_subset = set(current_subset)
                new_subset.remove(swap_out)
                new_score = evaluate(list(new_subset))
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_subset = new_subset
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_subset = list(current_subset)
            else:
                if len(current_subset) < n and random.random() < 0.2:
                    swap_in = random.choice(out_list)
                    new_subset = set(current_subset)
                    new_subset.add(swap_in)
                    new_score = evaluate(list(new_subset))
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_subset = new_subset
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_subset = list(current_subset)
                else:
                    swap_in = random.choice(out_list)
                    swap_out = random.choice(in_list)
                    new_subset = set(current_subset)
                    new_subset.remove(swap_out)
                    new_subset.add(swap_in)
                    new_score = evaluate(list(new_subset))
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_subset = new_subset
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_subset = list(current_subset)
            
            T *= cooling
            if T < min_T:
                break
        
        current_subset, current_score = local_search(current_subset)
        if current_score > best_score:
            best_score = current_score
            best_subset = list(current_subset)
    
    return list(best_subset)
