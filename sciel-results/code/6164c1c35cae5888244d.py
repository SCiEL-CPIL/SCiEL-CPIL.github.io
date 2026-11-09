import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    # Build conflict graph
    node_set = set(valid_nodes)
    conflicts = {v: set() for v in valid_nodes}
    for i_idx, v in enumerate(valid_nodes):
        for u in valid_nodes[i_idx + 1:]:
            if v + u in squares:
                conflicts[v].add(u)
                conflicts[u].add(v)
    
    def is_independent(v, chosen):
        return not (conflicts[v] & chosen)
        
    def greedy_mis(order):
        chosen = set()
        for v in order:
            if is_independent(v, chosen):
                chosen.add(v)
        return chosen

    def local_search(chosen, max_iters=200):
        chosen = set(chosen)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # 1. Try adding nodes
            for v in valid_nodes:
                if v in chosen:
                    continue
                if is_independent(v, chosen):
                    chosen.add(v)
                    improved = True
            
            # 2. Try 1-Swaps
            for v in valid_nodes:
                if v in chosen:
                    continue
                conf_in_chosen = conflicts[v] & chosen
                if len(conf_in_chosen) == 1:
                    u = next(iter(conf_in_chosen))
                    chosen.discard(u)
                    chosen.add(v)
                    improved = True
            
            # 3. Try 2-Swaps: remove 2 nodes, add 2 nodes
            if time.time() > deadline:
                break
            for v in valid_nodes:
                if v in chosen:
                    continue
                conf_in_chosen = conflicts[v] & chosen
                if len(conf_in_chosen) == 2:
                    conf_list = list(conf_in_chosen)
                    u1, u2 = conf_list[0], conf_list[1]
                    # Remove u1, u2; add v
                    # Now check if we can add another node w that was blocked by u1 or u2
                    temp_chosen = chosen - {u1, u2}
                    temp_chosen.add(v)
                    added = False
                    for w in valid_nodes:
                        if w in temp_chosen:
                            continue
                        if is_independent(w, temp_chosen):
                            temp_chosen.add(w)
                            added = True
                            break
                    if added:
                        chosen = temp_chosen
                        improved = True
                        break  # Restart the iteration since chosen changed significantly
        
        return chosen

    def perturbation(chosen, k):
        # Remove k random nodes
        chosen = set(chosen)
        to_remove = random.sample(list(chosen), min(k, len(chosen)))
        for u in to_remove:
            chosen.discard(u)
        # Re-add greedily from the removed + some random others?
        # Just run local search again on the perturbed set
        return local_search(chosen, max_iters=50)

    def targeted_perturbation(chosen):
        """Remove nodes with high degree in the induced subgraph, then re-optimize"""
        chosen = set(chosen)
        if not chosen:
            return chosen
        
        # Compute degree within chosen
        degrees = {}
        for v in chosen:
            degrees[v] = len(conflicts[v] & chosen)
        
        # Remove the 2-4 highest-degree nodes
        sorted_by_deg = sorted(chosen, key=lambda x: -degrees[x])
        k = random.randint(2, min(4, len(chosen)))
        to_remove = sorted_by_deg[:k]
        
        for u in to_remove:
            chosen.discard(u)
        
        return local_search(chosen, max_iters=80)

    best = set()
    random.seed(42)
    
    # Initial strategies
    orders = [
        valid_nodes,
        sorted(valid_nodes, key=lambda x: len(conflicts[x])), # Degree ascending
        sorted(valid_nodes, key=lambda x: -len(conflicts[x])), # Degree descending
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=100)
        if len(chosen) > len(best):
            best = chosen

    # Iterated Local Search
    # Start from best or random
    current = set(best)
    if not current:
        current = greedy_mis(valid_nodes)
        
    for i in range(150): # Max restarts
        if time.time() > deadline:
            break
        
        # Alternate between random and targeted perturbation
        if i % 3 == 0:
            k = random.randint(3, 10)
            current = perturbation(current, k)
        elif i % 3 == 1:
            current = targeted_perturbation(current)
        else:
            # Random restart with different order
            random_order = valid_nodes[:]
            random.shuffle(random_order)
            cand = greedy_mis(random_order)
            cand = local_search(cand, max_iters=60)
            if len(cand) > len(best):
                best = cand
            if len(cand) > len(current):
                current = cand
            continue
        
        if len(current) > len(best):
            best = current
            
        # Occasionally restart from scratch with random order
        if i % 15 == 0 and i > 0:
            random_order = valid_nodes[:]
            random.shuffle(random_order)
            cand = greedy_mis(random_order)
            cand = local_search(cand, max_iters=50)
            if len(cand) > len(best):
                best = cand
            if len(cand) > len(current):
                current = cand

    if not best:
        best = {valid_nodes[0]}
        
    return list(best)
