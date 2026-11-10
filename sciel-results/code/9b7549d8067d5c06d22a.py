import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # 1. Generate odds
    odds = [i for i in range(1, 257) if i % 2 != 0]
    
    # 2. Generate candidate evens (those that don't conflict with any odd)
    evens = [i for i in range(1, 257) if i % 2 == 0]
    
    def conflicts(a, b):
        return (a * b) % (a + b) == 0
    
    cand_evens = []
    for e in evens:
        valid = True
        for o in odds:
            if conflicts(e, o):
                valid = False
                break
        if valid:
            cand_evens.append(e)
            
    n = len(cand_evens)
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if conflicts(cand_evens[i], cand_evens[j]):
                adj[i][j] = True
                adj[j][i] = True
    
    # Precompute neighbor sets for fast lookup
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j]:
                neighbors[i].add(j)
    
    def greedy_is(order):
        """Greedy independent set given vertex order."""
        selected = set()
        for v in order:
            if all(v not in neighbors[u] for u in selected):
                selected.add(v)
        return selected
    
    def local_search(selected):
        """1-for-1 and 1-for-2 swap local search."""
        selected = set(selected)
        improved = True
        while improved:
            improved = False
            # Try to add any unselected node
            for v in range(n):
                if v not in selected:
                    if all(v not in neighbors[u] for u in selected):
                        selected.add(v)
                        improved = True
            # Try 1-for-2 swaps: remove one, try to add two
            for v in list(selected):
                trial = selected - {v}
                added = 0
                for w in range(n):
                    if w not in trial and w != v:
                        if all(w not in neighbors[u] for u in trial):
                            trial.add(w)
                            added += 1
                            if added >= 2:
                                break
                if len(trial) > len(selected):
                    selected = trial
                    improved = True
        return selected
    
    # Multi-start: try different orderings
    best_is = set()
    
    # Ordering 1: by degree (ascending)
    deg_order = sorted(range(n), key=lambda v: len(neighbors[v]))
    is1 = local_search(greedy_is(deg_order))
    if len(is1) > len(best_is):
        best_is = is1
    
    # Ordering 2: by value (ascending)
    val_order = sorted(range(n), key=lambda v: cand_evens[v])
    is2 = local_search(greedy_is(val_order))
    if len(is2) > len(best_is):
        best_is = is2
    
    # Ordering 3: reverse degree
    rdeg_order = sorted(range(n), key=lambda v: -len(neighbors[v]))
    is3 = local_search(greedy_is(rdeg_order))
    if len(is3) > len(best_is):
        best_is = is3
    
    # Try several random orderings if time permits
    if time.time() < deadline - 5.0 and n <= 40:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            rand_order = list(range(n))
            random.shuffle(rand_order)
            isr = local_search(greedy_is(rand_order))
            if len(isr) > len(best_is):
                best_is = isr
    
    # For larger n, use a better branch-and-bound MIS
    if n > 30 and time.time() < deadline - 3.0:
        # Branch and bound with coloring upper bound
        def branch_and_bound(vertices, current_size, current_set):
            if time.time() > deadline - 2.0:
                return current_set
            if not vertices:
                return current_set
            
            # Upper bound: current_size + len(vertices)
            if current_size + len(vertices) <= len(best_is):
                return current_set
            
            # Pick vertex with max degree in subgraph
            v = max(vertices, key=lambda x: sum(1 for y in vertices if adj[x][y]))
            rest = [u for u in vertices if u != v]
            
            # Include v
            non_neighbors = [u for u in rest if not adj[v][u]]
            sol_incl = branch_and_bound(non_neighbors, current_size + 1, current_set | {v})
            
            # Exclude v
            sol_excl = branch_and_bound(rest, current_size, current_set)
            
            if len(sol_incl) >= len(sol_excl):
                return sol_incl
            return sol_excl
        
        bb_result = branch_and_bound(list(range(n)), 0, set())
        if len(bb_result) > len(best_is):
            best_is = bb_result
    
    final_evens = [cand_evens[i] for i in sorted(best_is)]
    result = odds + final_evens
    
    return result
