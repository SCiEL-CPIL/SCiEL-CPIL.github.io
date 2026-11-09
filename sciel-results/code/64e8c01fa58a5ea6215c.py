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
            # We can optimize by only checking nodes that might be addable
            # But N is small, just check all
            for v in valid_nodes:
                if v in chosen:
                    continue
                if is_independent(v, chosen):
                    chosen.add(v)
                    improved = True
            
            # 2. Try 1-Swaps
            # For each v not in chosen, see if we can remove one conflict and add v
            for v in valid_nodes:
                if v in chosen:
                    continue
                conf_in_chosen = conflicts[v] & chosen
                if not conf_in_chosen:
                    # Should have been added in step 1, but maybe added in this loop?
                    # No, step 1 finished. If it's independent, it was added.
                    # If we are here, it must have conflicts.
                    pass
                elif len(conf_in_chosen) == 1:
                    u = next(iter(conf_in_chosen))
                    # Check if removing u allows v
                    # v conflicts with chosen - {u} ?
                    # conf_in_chosen was {u}. So v conflicts with NO OTHER in chosen.
                    # So yes, we can swap.
                    chosen.discard(u)
                    chosen.add(v)
                    improved = True
                elif len(conf_in_chosen) > 1:
                    # Try 2-Swap? 
                    # For each pair in conf_in_chosen, remove both, add v? No, that reduces size.
                    # We need to add v and remove k nodes, and add k other nodes?
                    # For now, stick to 1-swap for speed, maybe add simple 2-swap if time permits.
                    pass
            
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
        
    for i in range(100): # Max restarts
        if time.time() > deadline:
            break
        
        # Perturb
        k = random.randint(3, 10)
        current = perturbation(current, k)
        
        if len(current) > len(best):
            best = current
            
        # Occasionally restart from scratch with random order
        if i % 10 == 0:
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
