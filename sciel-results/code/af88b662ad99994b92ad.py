import random
import time

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    d = 7
    n = 2 ** d
    
    # Precompute all edges
    all_edges = []
    edge_to_bit = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
                edge_to_bit[(x, y)] = i
    
    num_edges = len(all_edges)  # 448
    
    def can_add(chosen_set, edge):
        """Check if edge can be added without forming C4."""
        u, v = edge
        i = edge_to_bit[edge]
        for j in range(d):
            if j == i:
                continue
            w1 = u ^ (1 << j)
            w2 = v ^ (1 << j)
            e1 = (min(u, w1), max(u, w1))
            e2 = (min(v, w2), max(v, w2))
            e3 = (min(w1, w2), max(w1, w2))
            if e1 in chosen_set and e2 in chosen_set and e3 in chosen_set:
                return False
        return True
    
    def greedy_random(seed):
        random.seed(seed)
        chosen = set()
        order = list(range(num_edges))
        random.shuffle(order)
        
        for idx in order:
            if time.time() - start_time > time_limit:
                break
            edge = all_edges[idx]
            if can_add(chosen, edge):
                chosen.add(edge)
        
        return chosen
    
    # Run multiple greedy trials
    best = set()
    num_trials = 0
    max_trials_time = time_limit * 0.5
    while time.time() - start_time < start_time + max_trials_time:
        trial = greedy_random(seed=12345 + num_trials)
        if len(trial) > len(best):
            best = trial
        num_trials += 1
    
    # Local search
    def local_search(chosen, time_budget):
        if time.time() - start_time > time_budget:
            return chosen
        
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            edges_list = list(chosen)
            random.shuffle(edges_list)
            
            for rem_edge in edges_list:
                if time.time() - start_time > time_budget:
                    break
                
                new_chosen = set(chosen)
                new_chosen.discard(rem_edge)
                
                added_count = 0
                for idx in range(num_edges):
                    if time.time() - start_time > time_budget:
                        break
                    edge = all_edges[idx]
                    if edge not in new_chosen and can_add(new_chosen, edge):
                        new_chosen.add(edge)
                        added_count += 1
                
                if len(new_chosen) > len(chosen):
                    chosen = new_chosen
                    improved = True
                    break
        
        return chosen
    
    # Run local search with time budget
    ls_budget = time_limit * 0.85
    best = local_search(best, ls_budget)
    
    # Try more local search if time permits
    if time.time() - start_time < time_limit * 0.95:
        best = local_search(best, time_limit * 0.97)
    
    out = [[u, v] for u, v in best]
    return out
