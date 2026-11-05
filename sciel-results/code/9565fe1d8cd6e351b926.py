"""Combined program: B&B exact solver + bitmask local search + smooth polish."""
import time
import random
import numpy as np

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj, edges

def _bron_kerbosch_maximal_cliques(adj, n):
    cliques = []
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        P_union_X = P | X
        if not P_union_X:
            return
        best_pivot = None
        max_count = -1
        for u in P_union_X:
            count = len(adj[u] & P)
            if count > max_count:
                max_count = count
                best_pivot = u
        if best_pivot is None:
            return
        candidates = P - adj[best_pivot]
        for v in list(candidates):
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R_new, P_new, X_new)
            P.remove(v)
            X.add(v)
    bk(set(), set(range(n)), set())
    return cliques

def _greedy_init(relevant_cliques, vertex_to_cliques, n):
    """Greedy set cover using bitmask coverage."""
    num_cliques = len(relevant_cliques)
    if num_cliques == 0:
        return set()
    uncovered_mask = (1 << num_cliques) - 1
    selected = set()
    while uncovered_mask:
        best_vertex = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = 0
            for ci in vertex_to_cliques[v]:
                if uncovered_mask & (1 << ci):
                    count += 1
            if count > best_count:
                best_count = count
                best_vertex = v
        if best_vertex is None or best_count <= 0:
            break
        selected.add(best_vertex)
        for ci in vertex_to_cliques[best_vertex]:
            uncovered_mask &= ~(1 << ci)
    return selected

def _prune(selected, vertex_to_cliques, num_cliques):
    """Remove redundant vertices."""
    selected = set(selected)
    if not selected:
        return selected
    cover_count = [0] * num_cliques
    for v in selected:
        for ci in vertex_to_cliques[v]:
            cover_count[ci] += 1
    changed = True
    while changed:
        changed = False
        for v in list(selected):
            can_remove = True
            for ci in vertex_to_cliques[v]:
                if cover_count[ci] == 1:
                    can_remove = False
                    break
            if can_remove:
                selected.remove(v)
                for ci in vertex_to_cliques[v]:
                    cover_count[ci] -= 1
                changed = True
    return selected

def _local_search(selected, vertex_to_cliques, num_cliques, n, deadline, rng):
    """Bitmask-based local search with removals and 2-for-1 swaps."""
    current_S = set(selected)
    if not current_S:
        return current_S
    cover_count = [0] * num_cliques
    for v in current_S:
        for ci in vertex_to_cliques[v]:
            cover_count[ci] += 1
    
    no_improve = 0
    max_no_improve = 5000
    step = 0
    while time.time() < deadline and no_improve < max_no_improve:
        step += 1
        # Try removals
        removed = False
        for v_out in list(current_S):
            is_critical = False
            for ci in vertex_to_cliques[v_out]:
                if cover_count[ci] == 1:
                    is_critical = True
                    break
            if not is_critical:
                current_S.remove(v_out)
                for ci in vertex_to_cliques[v_out]:
                    cover_count[ci] -= 1
                no_improve = 0
                removed = True
                break
        if removed:
            continue
        
        # Try 2-for-1: replace two vertices with one
        if len(current_S) >= 2:
            vertices = list(current_S)
            v1, v2 = rng.sample(vertices, 2)
            # Find cliques uniquely covered by v1 or v2
            others = current_S - {v1, v2}
            must_cover = set()
            for ci in vertex_to_cliques[v1]:
                if cover_count[ci] == 1:  # only v1 covers it
                    must_cover.add(ci)
            for ci in vertex_to_cliques[v2]:
                if cover_count[ci] == 1:  # only v2 covers it
                    must_cover.add(ci)
            # Also cliques covered by exactly one of v1,v2 and no other
            for ci in vertex_to_cliques[v1]:
                if cover_count[ci] == 2 and ci in vertex_to_cliques[v2]:
                    # covered by v1 and v2, check if any other
                    if not any(ci in vertex_to_cliques[w] for w in others):
                        must_cover.add(ci)
            for ci in vertex_to_cliques[v2]:
                if cover_count[ci] == 2 and ci in vertex_to_cliques[v1]:
                    if not any(ci in vertex_to_cliques[w] for w in others):
                        must_cover.add(ci)
            
            if must_cover:
                must_cover_frozen = frozenset(must_cover)
                for w in range(n):
                    if w in current_S:
                        continue
                    if must_cover_frozen <= frozenset(vertex_to_cliques[w]):
                        current_S.discard(v1)
                        current_S.discard(v2)
                        for ci in vertex_to_cliques[v1]:
                            cover_count[ci] -= 1
                        for ci in vertex_to_cliques[v2]:
                            cover_count[ci] -= 1
                        current_S.add(w)
                        for ci in vertex_to_cliques[w]:
                            cover_count[ci] += 1
                        no_improve = 0
                        break
                else:
                    no_improve += 1
            else:
                # Can remove both
                current_S.discard(v1)
                current_S.discard(v2)
                for ci in vertex_to_cliques[v1]:
                    cover_count[ci] -= 1
                for ci in vertex_to_cliques[v2]:
                    cover_count[ci] -= 1
                no_improve = 0
        else:
            no_improve += 1
    return current_S

def _branch_and_bound(relevant_cliques, vertex_to_cliques, n, deadline, initial_bound):
    """Exact B&B for minimum hitting set. Returns optimal set or best found."""
    num_cliques = len(relevant_cliques)
    if num_cliques == 0:
        return set()
    
    # Sort cliques by size (smaller first for better pruning)
    clique_order = sorted(range(num_cliques), key=lambda i: len(relevant_cliques[i]))
    
    best_solution = set()
    best_size = initial_bound
    
    # Precompute for each vertex, the bitmask of cliques it covers
    vertex_masks = [0] * n
    for v in range(n):
        mask = 0
        for ci in vertex_to_cliques[v]:
            mask |= (1 << ci)
        vertex_masks[v] = mask
    
    all_mask = (1 << num_cliques) - 1
    
    def bb(selected_mask, covered_mask, remaining_cliques, current_size):
        nonlocal best_solution, best_size
        if time.time() > deadline:
            return
        if current_size >= best_size:
            return
        if covered_mask == all_mask:
            best_size = current_size
            best_solution = set(v for v in range(n) if selected_mask & (1 << v))
            return
        # Lower bound: number of remaining cliques / max cliques per vertex
        if not remaining_cliques:
            return
        max_cover = 0
        for v in range(n):
            if selected_mask & (1 << v):
                continue
            cover = bin(remaining_cliques & vertex_masks[v]).count('1')
            if cover > max_cover:
                max_cover = cover
        if max_cover == 0:
            return
        lb = (len(bin(remaining_cliques).count('1') - 1) // max_cover) + 1
        if current_size + lb >= best_size:
            return
        
        # Pick the most constrained remaining clique
        # Find a remaining clique with fewest candidate vertices
        best_clique = None
        min_candidates = n + 1
        for ci in range(num_cliques):
            if not (remaining_cliques & (1 << ci)):
                continue
            candidates = 0
            for v in range(n):
                if selected_mask & (1 << v):
                    continue
                if vertex_masks[v] & (1 << ci):
                    candidates += 1
            if candidates < min_candidates:
                min_candidates = candidates
                best_clique = ci
            if candidates <= 1:
                break
        
        if best_clique is None:
            return
        
        # Branch on vertices that cover best_clique
        candidates = [v for v in range(n) 
                      if not (selected_mask & (1 << v)) 
                      and (vertex_masks[v] & (1 << best_clique))]
        # Sort candidates by number of cliques they cover (more first)
        candidates.sort(key=lambda v: -bin(vertex_masks[v] & remaining_cliques).count('1'))
        
        for v in candidates:
            new_selected = selected_mask | (1 << v)
            new_covered = covered_mask | vertex_masks[v]
            new_remaining = remaining_cliques & ~vertex_masks[v]
            bb(new_selected, new_covered, new_remaining, current_size + 1)
            if time.time() > deadline:
                return
    
    bb(0, 0, all_mask, 0)
    return best_solution

def _smooth_polish(selected, relevant_cliques, vertex_to_cliques, n, deadline, rng):
    """Final polish using soft-min surrogate to escape local optima."""
    current_S = set(selected)
    if not current_S:
        return current_S
    num_cliques = len(relevant_cliques)
    best_S = set(current_S)
    best_size = len(current_S)
    
    cover_count = [0] * num_cliques
    for v in current_S:
        for ci in vertex_to_cliques[v]:
            cover_count[ci] += 1
    
    iterations = 0
    max_iter = 500
    while time.time() < deadline and iterations < max_iter:
        iterations += 1
        # Soft-min: weight cliques by 1/(cover_count+epsilon) to prioritize scarce coverage
        epsilon = 0.1
        weights = [1.0 / (cover_count[ci] + epsilon) for ci in range(num_cliques)]
        
        # Try swap: remove one, add one that improves soft objective
        improved = False
        for v_out in list(current_S):
            # Compute soft objective loss from removing v_out
            loss = 0.0
            for ci in vertex_to_cliques[v_out]:
                if cover_count[ci] == 1:
                    loss += 1000.0  # Hard constraint violation penalty
                else:
                    loss += weights[ci]
            
            # Find best v_in to add
            best_gain = -1
            best_v_in = None
            for v_in in range(n):
                if v_in in current_S:
                    continue
                gain = 0.0
                for ci in vertex_to_cliques[v_in]:
                    if cover_count[ci] == 0:
                        gain += 1000.0
                    else:
                        gain += weights[ci] * 0.5  # Diminishing returns
                if gain > best_gain:
                    best_gain = gain
                    best_v_in = v_in
            
            if best_v_in is not None and (best_gain - loss) > 0.1:
                # Accept swap
                current_S.remove(v_out)
                for ci in vertex_to_cliques[v_out]:
                    cover_count[ci] -= 1
                current_S.add(best_v_in)
                for ci in vertex_to_cliques[best_v_in]:
                    cover_count[ci] += 1
                if len(current_S) < best_size:
                    best_S = set(current_S)
                    best_size = len(current_S)
                improved = True
                break
        
        if not improved:
            break
    
    return best_S if len(best_S) <= len(current_S) else current_S

def _solve_one_instance(n, p, seed, time_limit=10.0):
    adj, edges = _generate_graph(n, p, seed)
    all_cliques = _bron_kerbosch_maximal_cliques(adj, n)
    relevant_cliques = [c for c in all_cliques if len(c) >= 2]
    
    if not relevant_cliques:
        return []
    
    num_cliques = len(relevant_cliques)
    
    # Precompute vertex -> clique indices
    vertex_to_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(relevant_cliques):
        for v in clique:
            vertex_to_cliques[v].append(i)
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Phase 1: Greedy initialization
    greedy_S = _greedy_init(relevant_cliques, vertex_to_cliques, n)
    greedy_S = _prune(greedy_S, vertex_to_cliques, num_cliques)
    
    # Phase 2: B&B with time budget (use greedy size as initial bound)
    bb_deadline = start_time + min(0.4 * time_limit, 5.0)
    bb_result = _branch_and_bound(relevant_cliques, vertex_to_cliques, n, bb_deadline, len(greedy_S) + 1)
    
    # Phase 3: Local search on best of greedy and B&B
    rng = random.Random(seed + 42)
    current_S = bb_result if bb_result else greedy_S
    if not current_S or len(current_S) > len(greedy_S):
        current_S = greedy_S
    
    ls_deadline = start_time + min(0.7 * time_limit, 10.0)
    current_S = _local_search(current_S, vertex_to_cliques, num_cliques, n, ls_deadline, rng)
    current_S = _prune(current_S, vertex_to_cliques, num_cliques)
    
    # Phase 4: Smooth polish
    polish_deadline = deadline - 0.5
    current_S = _smooth_polish(current_S, relevant_cliques, vertex_to_cliques, n, polish_deadline, rng)
    current_S = _prune(current_S, vertex_to_cliques, num_cliques)
    
    return sorted(current_S)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    time_budget = 130.0
    total_n2 = sum(n * n for n, p, seed in instances)
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time()
        if remaining_time < 5.0:
            for _ in range(idx, len(instances)):
                results.append([])
            break
        
        n2 = n * n
        allocated = min(remaining_time - 5.0, time_budget * n2 / total_n2)
        allocated = max(allocated, 3.0)
        
        try:
            result = _solve_one_instance(n, p, seed, time_limit=allocated)
            results.append(result)
        except Exception:
            results.append(list(range(n)))
    
    return results
