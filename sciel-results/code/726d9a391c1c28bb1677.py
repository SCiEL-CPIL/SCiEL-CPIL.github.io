import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety (evaluator kills at 90s)
    
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
    
    # Precompute degrees for ordering
    degrees = {v: len(conflicts[v]) for v in valid_nodes}
    
    def is_independent(v, chosen):
        return not (conflicts[v] & chosen)
        
    def greedy_mis(order):
        chosen = set()
        for v in order:
            if is_independent(v, chosen):
                chosen.add(v)
        return chosen

    def local_search(chosen, max_iters=500):
        chosen = set(chosen)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # 1. Try adding nodes (sorted by degree ascending for better choices)
            candidates = [v for v in valid_nodes if v not in chosen and is_independent(v, chosen)]
            candidates.sort(key=lambda x: len(conflicts[x]))
            for v in candidates:
                if is_independent(v, chosen):
                    chosen.add(v)
                    improved = True
            
            # 2. Try 1-Swaps: remove one conflict, add v
            for v in valid_nodes:
                if v in chosen:
                    continue
                conf_in_chosen = conflicts[v] & chosen
                if len(conf_in_chosen) == 1:
                    u = next(iter(conf_in_chosen))
                    # Check if removing u allows adding v and potentially more
                    chosen.discard(u)
                    chosen.add(v)
                    improved = True
                elif len(conf_in_chosen) == 2:
                    # Try 2-Swap: remove two conflicts, add v
                    # Only beneficial if we can add back at least 2 other nodes
                    conf_list = list(conf_in_chosen)
                    u1, u2 = conf_list[0], conf_list[1]
                    
                    # Temporarily remove u1, u2 and add v
                    chosen.discard(u1)
                    chosen.discard(u2)
                    chosen.add(v)
                    
                    # Try to add back any nodes that were blocked by u1 or u2
                    added_back = 0
                    for w in valid_nodes:
                        if w in chosen:
                            continue
                        if w == v:
                            continue
                        if is_independent(w, chosen):
                            chosen.add(w)
                            added_back += 1
                        if time.time() > deadline:
                            break
                    
                    # If we added back at least as many as we removed, keep the change
                    if added_back >= 1:
                        improved = True
                    else:
                        # Revert
                        chosen.discard(v)
                        chosen.add(u1)
                        chosen.add(u2)
            
        return chosen

    def simulated_annealing(initial_set, max_iters=5000):
        """Simulated annealing on the independent set problem.
        
        State: a set S of nodes.
        Moves:
          - Add a node v not in S (if independent)
          - Remove a node v from S
          - Swap: remove v from S, add u not in S (u must be independent of S\{v})
        
        Objective: maximize |S| while maintaining independence.
        We use a penalty-based approach: if a move breaks independence, we accept
        with lower probability. Actually, let's keep it simpler: only allow valid
        moves (maintain independence), and use SA to decide when to accept a
        temporarily smaller set to escape local optima.
        """
        chosen = set(initial_set)
        best = set(chosen)
        best_size = len(chosen)
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iters - 1))
        T = T_start
        
        rng = random.Random(42)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            T *= cooling
            
            # Generate a random valid move
            move_type = rng.random()
            
            if move_type < 0.4 and len(chosen) < len(valid_nodes):
                # Try adding a random node
                candidates = [v for v in valid_nodes if v not in chosen]
                if candidates:
                    v = rng.choice(candidates)
                    if is_independent(v, chosen):
                        new_set = set(chosen)
                        new_set.add(v)
                        # Always accept if it improves
                        if len(new_set) >= len(chosen):
                            chosen = new_set
                            if len(chosen) > best_size:
                                best = set(chosen)
                                best_size = len(chosen)
                        else:
                            # Accept worse with probability exp(-delta/T)
                            delta = len(chosen) - len(new_set)  # positive means worse
                            if delta > 0 and rng.random() < math.exp(-delta / max(T, 1e-10)):
                                chosen = new_set
                    else:
                        # v conflicts with chosen; try swap: remove one conflict, add v
                        conf_in_chosen = conflicts[v] & chosen
                        if conf_in_chosen:
                            u = rng.choice(list(conf_in_chosen))
                            new_set = set(chosen)
                            new_set.discard(u)
                            new_set.add(v)
                            # Check if new_set is independent
                            if all(is_independent(x, new_set - {x}) for x in [v]):
                                if len(new_set) >= len(chosen):
                                    chosen = new_set
                                    if len(chosen) > best_size:
                                        best = set(chosen)
                                        best_size = len(chosen)
                                else:
                                    delta = len(chosen) - len(new_set)
                                    if delta > 0 and rng.random() < math.exp(-delta / max(T, 1e-10)):
                                        chosen = new_set
                                    
            elif move_type < 0.7 and chosen:
                # Try removing a random node
                v = rng.choice(list(chosen))
                new_set = set(chosen)
                new_set.discard(v)
                # Removing always makes it smaller or same; only accept if we can re-add
                # Actually, just accept with probability to explore
                if rng.random() < math.exp(1.0 / max(T, 1e-10)):
                    chosen = new_set
                else:
                    # Try to re-add some nodes after removal
                    for w in valid_nodes:
                        if w not in chosen and is_independent(w, chosen):
                            if rng.random() < 0.5:  # probabilistic re-add
                                chosen.add(w)
                    if len(chosen) > best_size:
                        best = set(chosen)
                        best_size = len(chosen)
                        
            else:
                # Swap: remove v from chosen, add u not in chosen
                if chosen and len(chosen) < len(valid_nodes):
                    v = rng.choice(list(chosen))
                    candidates = [w for w in valid_nodes if w not in chosen]
                    if candidates:
                        u = rng.choice(candidates)
                        # Check if u is independent of chosen \ {v}
                        temp_chosen = set(chosen)
                        temp_chosen.discard(v)
                        if is_independent(u, temp_chosen):
                            new_set = temp_chosen
                            new_set.add(u)
                            if len(new_set) >= len(chosen):
                                chosen = new_set
                                if len(chosen) > best_size:
                                    best = set(chosen)
                                    best_size = len(chosen)
                            else:
                                delta = len(chosen) - len(new_set)
                                if delta > 0 and rng.random() < math.exp(-delta / max(T, 1e-10)):
                                    chosen = new_set
        
        return best

    best = set()
    random.seed(42)
    
    # Initial strategies with more diverse orderings
    orders = [
        valid_nodes,
        sorted(valid_nodes, key=lambda x: len(conflicts[x])), # Degree ascending
        sorted(valid_nodes, key=lambda x: -len(conflicts[x])), # Degree descending
        sorted(valid_nodes, key=lambda x: (len(conflicts[x]), x)), # Degree then value
        sorted(valid_nodes, key=lambda x: (-len(conflicts[x]), x)), # Negative degree then value
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=300)
        if len(chosen) > len(best):
            best = chosen

    # Simulated Annealing phase - the main improvement
    # Run SA from multiple starting points
    if time.time() < deadline - 10:
        sa_starts = [
            set(best),
            greedy_mis(valid_nodes),
            greedy_mis(sorted(valid_nodes, key=lambda x: len(conflicts[x]))),
            greedy_mis(sorted(valid_nodes, key=lambda x: -len(conflicts[x]))),
        ]
        
        # Add some random starts
        for _ in range(3):
            random_order = valid_nodes[:]
            random.shuffle(random_order)
            sa_starts.append(greedy_mis(random_order))
        
        for start_idx, start_set in enumerate(sa_starts):
            if time.time() > deadline - 5:
                break
            
            # Run SA with different iteration budgets based on remaining time
            remaining = deadline - time.time()
            if remaining > 20:
                max_sa_iters = 8000
            elif remaining > 10:
                max_sa_iters = 4000
            else:
                max_sa_iters = 2000
            
            sa_result = simulated_annealing(start_set, max_iters=max_sa_iters)
            
            # Polish with local search
            sa_result = local_search(sa_result, max_iters=200)
            
            if len(sa_result) > len(best):
                best = sa_result
    
    # Final polish: try to improve the best solution with more iterations
    if time.time() < deadline - 5:
        polished = local_search(best, max_iters=2000)
        if len(polished) > len(best):
            best = polished
    
    # One more final attempt with a different strategy
    if time.time() < deadline - 3 and best:
        # Try removing a few random nodes from best and re-optimizing
        for attempt in range(10):
            if time.time() > deadline - 2:
                break
            trial = set(best)
            to_remove = random.sample(list(trial), min(3, len(trial)))
            for u in to_remove:
                trial.discard(u)
            trial = local_search(trial, max_iters=100)
            if len(trial) > len(best):
                best = trial

    if not best:
        best = {valid_nodes[0]}
        
    return list(best)
