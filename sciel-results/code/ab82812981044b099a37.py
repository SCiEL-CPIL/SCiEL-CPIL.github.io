"""Combined program: Bron-Kerbosch + greedy set cover + local search + branch-and-bound polish."""
import time
import random

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        U = P | X
        if not U:
            return
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _validate_solution(selected, cliques):
    """Check if selected vertices meet every maximal clique of size >= 2."""
    selected_set = set(selected)
    for clique in cliques:
        if not (selected_set & clique):
            return False
    return True

def _greedy_set_cover(cliques):
    """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
    if not cliques:
        return []
    
    n_cliques = len(cliques)
    # Build vertex -> cliques mapping
    vertex_to_cliques = {}
    for i, clique in enumerate(cliques):
        for v in clique:
            if v not in vertex_to_cliques:
                vertex_to_cliques[v] = []
            vertex_to_cliques[v].append(i)
    
    covered = [False] * n_cliques
    selected = []
    uncovered_count = n_cliques
    
    while uncovered_count > 0:
        best_v = None
        best_count = -1
        for v, clist in vertex_to_cliques.items():
            count = sum(1 for idx in clist if not covered[idx])
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count == 0:
            break
        selected.append(best_v)
        for idx in vertex_to_cliques[best_v]:
            if not covered[idx]:
                covered[idx] = True
                uncovered_count -= 1
    
    return sorted(selected)

def _local_search(adj, initial_solution, cliques, max_iters=5000, time_budget=5.0):
    """Optimized local search using coverage counts for fast validity checks."""
    start = time.time()
    current = list(initial_solution)
    n = len(adj)
    
    if not cliques:
        return sorted(current)
    
    n_cliques = len(cliques)
    
    # Precompute vertex -> list of clique indices
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Initialize coverage counts
    cover_count = [0] * n_cliques
    current_set = set(current)
    for v in current_set:
        for ci in vertex_cliques[v]:
            cover_count[ci] += 1
    
    for _ in range(max_iters):
        if time.time() - start > time_budget:
            break
        improved = False
        current_set = set(current)
        
        # 1. Try simple removals
        for v in list(current):
            safe_to_remove = True
            for ci in vertex_cliques[v]:
                if cover_count[ci] == 1:
                    safe_to_remove = False
                    break
            if safe_to_remove:
                for ci in vertex_cliques[v]:
                    cover_count[ci] -= 1
                current_set.remove(v)
                current = list(current_set)
                improved = True
                break
        
        if improved:
            continue
        
        # 2. Try 1-1 swaps
        for v in list(current):
            deficit_cliques = []
            for ci in vertex_cliques[v]:
                if cover_count[ci] == 1:
                    deficit_cliques.append(ci)
            
            if not deficit_cliques:
                for ci in vertex_cliques[v]:
                    cover_count[ci] -= 1
                current_set.remove(v)
                current = list(current_set)
                improved = True
                break
            
            if len(deficit_cliques) == 1:
                ci = deficit_cliques[0]
                candidates = [u for u in cliques[ci] if u not in current_set]
                if candidates:
                    # Pick candidate that covers most other cliques
                    best_u = max(candidates, key=lambda u: len(vertex_cliques[u]))
                    for c1 in vertex_cliques[v]:
                        cover_count[c1] -= 1
                    for c2 in vertex_cliques[best_u]:
                        cover_count[c2] += 1
                    current_set.remove(v)
                    current_set.add(best_u)
                    current = list(current_set)
                    improved = True
                    break
            else:
                potential_u = set(cliques[deficit_cliques[0]])
                for ci in deficit_cliques[1:]:
                    potential_u &= set(cliques[ci])
                potential_u -= current_set
                if potential_u:
                    # Pick best candidate by coverage
                    best_u = max(potential_u, key=lambda u: len(vertex_cliques[u]))
                    for c1 in vertex_cliques[v]:
                        cover_count[c1] -= 1
                    for c2 in vertex_cliques[best_u]:
                        cover_count[c2] += 1
                    current_set.remove(v)
                    current_set.add(best_u)
                    current = list(current_set)
                    improved = True
                    break
        
        if not improved:
            break
    
    return sorted(current)

def _packing_lower_bound(uncovered_cliques, vertex_to_cliques):
    """Fast packing-based lower bound: greedily select disjoint cliques.
    Each selected clique requires at least one new vertex not in the current solution.
    O(m^2) where m = number of uncovered cliques.
    """
    if not uncovered_cliques:
        return 0
    
    # Greedy packing: pick cliques that are vertex-disjoint
    used_vertices = set()
    count = 0
    # Sort by size (smaller cliques are easier to pack)
    sorted_uncov = sorted(uncovered_cliques, key=len)
    for clique in sorted_uncov:
        if clique & used_vertices:
            continue
        used_vertices |= clique
        count += 1
    return count

def _branch_and_bound(cliques, upper_bound, time_budget=10.0):
    """Branch-and-bound to find a set cover smaller than upper_bound.
    
    Returns a solution of size < upper_bound if found, else None.
    Uses packing lower bound for tighter pruning.
    """
    start = time.time()
    n_cliques = len(cliques)
    if n_cliques == 0:
        return []
    
    # Sort cliques by size (smallest first for better pruning)
    sorted_cliques = sorted(cliques, key=len)
    
    # Build vertex -> set of clique indices it covers
    vertex_to_cliques = {}
    for i, clique in enumerate(sorted_cliques):
        for v in clique:
            if v not in vertex_to_cliques:
                vertex_to_cliques[v] = set()
            vertex_to_cliques[v].add(i)
    
    use_bitmask = n_cliques <= 64
    if use_bitmask:
        all_mask = (1 << n_cliques) - 1
    else:
        all_mask = None
    
    best_found = None
    best_size = upper_bound
    nodes_visited = [0]
    
    def bbsolve(idx, selected, covered_mask, covered_set):
        nonlocal best_found, best_size
        if time.time() - start > time_budget:
            return
        
        nodes_visited[0] += 1
        
        if idx == n_cliques:
            if len(selected) < best_size:
                best_size = len(selected)
                best_found = list(selected)
            return
        
        # Check if current clique is already covered
        if use_bitmask:
            if (covered_mask >> idx) & 1:
                bbsolve(idx + 1, selected, covered_mask, covered_set)
                return
        else:
            if idx in covered_set:
                bbsolve(idx + 1, selected, covered_mask, covered_set)
                return
        
        # Packing lower bound: count remaining uncovered cliques
        if use_bitmask:
            remaining_mask = all_mask & ~(covered_mask) & ~((1 << idx) - 1)
            # Quick count of set bits
            remaining_count = bin(remaining_mask).count('1')
            # Tighter: packing bound
            if len(selected) + remaining_count >= best_size:
                return
            # Even tighter: try packing
            uncovered_list = []
            for ci in range(idx, n_cliques):
                if (remaining_mask >> ci) & 1:
                    uncovered_list.append(ci)
            if uncovered_list:
                lb = _packing_lower_bound([sorted_cliques[ci] for ci in uncovered_list], vertex_to_cliques)
                if len(selected) + lb >= best_size:
                    return
        else:
            remaining_count = 0
            uncovered_list = []
            for ci in range(idx, n_cliques):
                if ci not in covered_set:
                    remaining_count += 1
                    uncovered_list.append(ci)
            if len(selected) + remaining_count >= best_size:
                return
            if uncovered_list:
                lb = _packing_lower_bound([sorted_cliques[ci] for ci in uncovered_list], vertex_to_cliques)
                if len(selected) + lb >= best_size:
                    return
        
        # Try each vertex in the current clique
        clique = sorted_cliques[idx]
        
        # Score vertices by how many additional uncovered cliques they cover
        def vertex_score(v):
            if use_bitmask:
                additional = 0
                for ci in vertex_to_cliques.get(v, ()):
                    if ci > idx and (remaining_mask >> ci) & 1:
                        additional += 1
                return additional
            else:
                additional = 0
                for ci in vertex_to_cliques.get(v, ()):
                    if ci > idx and ci not in covered_set:
                        additional += 1
                return additional
        
        candidates = sorted(clique, key=vertex_score, reverse=True)
        
        for v in candidates:
            new_selected = selected + [v]
            if len(new_selected) >= best_size:
                continue
            
            if use_bitmask:
                new_mask = covered_mask
                for ci in vertex_to_cliques.get(v, ()):
                    new_mask |= (1 << ci)
                bbsolve(idx + 1, new_selected, new_mask, covered_set)
            else:
                new_set = covered_set | vertex_to_cliques.get(v, set())
                bbsolve(idx + 1, new_selected, covered_mask, new_set)
    
    bbsolve(0, [], 0, set())
    
    if best_found is not None:
        return best_found
    return None

def _random_local_search(adj, cliques, time_budget=5.0):
    """Generate random candidate solutions and improve via local search."""
    start = time.time()
    n = len(adj)
    rng = random.Random(42)
    
    if not cliques:
        return []
    
    best = None
    best_size = float('inf')
    
    # First try greedy
    greedy_sol = _greedy_set_cover(cliques)
    if _validate_solution(greedy_sol, cliques):
        best = greedy_sol
        best_size = len(greedy_sol)
    
    # Precompute vertex -> clique indices for fast coverage checks
    n_cliques = len(cliques)
    vertex_cliques = [set() for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].add(i)
    
    # Random restarts
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        # Start with a random subset biased towards smaller sizes
        max_subset = min(n, best_size + 1 if best else 5)
        subset_size = rng.randint(0, max_subset)
        candidate = set(rng.sample(range(n), subset_size))
        
        # Greedily add vertices to cover all cliques
        covered_cliques = set()
        for i, clique in enumerate(cliques):
            if candidate & clique:
                covered_cliques.add(i)
        
        while len(covered_cliques) < n_cliques:
            uncovered = [i for i in range(n_cliques) if i not in covered_cliques]
            if not uncovered:
                break
            
            # Pick vertex that covers most uncovered cliques
            best_v = None
            best_count = -1
            candidates = set()
            for i in uncovered:
                candidates.update(cliques[i])
            
            for v in candidates:
                count = len(vertex_cliques[v] & set(uncovered))
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            candidate.add(best_v)
            covered_cliques |= vertex_cliques[best_v]
        
        if len(covered_cliques) < n_cliques:
            continue
        
        if len(candidate) < best_size:
            improved = _local_search(adj, list(candidate), cliques, max_iters=1000, time_budget=0.3)
            if _validate_solution(improved, cliques) and len(improved) < best_size:
                best = improved
                best_size = len(improved)
    
    if best is None:
        return greedy_sol
    return best

def _tight_local_search(adj, initial_solution, cliques, time_budget=5.0):
    """Aggressive local search with 2-swap moves for final polish."""
    start = time.time()
    current = list(initial_solution)
    n = len(adj)
    
    if not cliques:
        return sorted(current)
    
    n_cliques = len(cliques)
    
    # Precompute vertex -> list of clique indices
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Initialize coverage counts
    cover_count = [0] * n_cliques
    current_set = set(current)
    for v in current_set:
        for ci in vertex_cliques[v]:
            cover_count[ci] += 1
    
    def is_valid():
        return all(c >= 1 for c in cover_count)
    
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        current_set = set(current)
        
        # 1. Try removals
        for v in list(current):
            safe = True
            for ci in vertex_cliques[v]:
                if cover_count[ci] == 1:
                    safe = False
                    break
            if safe:
                for ci in vertex_cliques[v]:
                    cover_count[ci] -= 1
                current_set.remove(v)
                current = list(current_set)
                improved = True
                break
        
        if improved:
            continue
        
        # 2. Try 1-1 swaps
        for v in list(current):
            deficit = [ci for ci in vertex_cliques[v] if cover_count[ci] == 1]
            if not deficit:
                for ci in vertex_cliques[v]:
                    cover_count[ci] -= 1
                current_set.remove(v)
                current = list(current_set)
                improved = True
                break
            
            # Find a vertex u that covers all deficit cliques
            potential = set(cliques[deficit[0]])
            for ci in deficit[1:]:
                potential &= set(cliques[ci])
            potential -= current_set
            
            if potential:
                best_u = max(potential, key=lambda u: len(vertex_cliques[u]))
                for c1 in vertex_cliques[v]:
                    cover_count[c1] -= 1
                for c2 in vertex_cliques[best_u]:
                    cover_count[c2] += 1
                current_set.remove(v)
                current_set.add(best_u)
                current = list(current_set)
                improved = True
                break
        
        # 3. Try 2-swap moves: remove two vertices, add two new ones
        if not improved and len(current) >= 2 and (time.time() - start) < time_budget - 0.5:
            cur_list = list(current)
            found_swap = False
            for i in range(len(cur_list)):
                if found_swap:
                    break
                for j in range(i + 1, len(cur_list)):
                    if found_swap:
                        break
                    v1, v2 = cur_list[i], cur_list[j]
                    # Find all cliques that would become uncovered
                    deficit_cliques = set()
                    for ci in vertex_cliques[v1]:
                        if cover_count[ci] == 1:
                            deficit_cliques.add(ci)
                    for ci in vertex_cliques[v2]:
                        if cover_count[ci] == 1:
                            deficit_cliques.add(ci)
                    
                    if not deficit_cliques:
                        # Both can be removed
                        for ci in vertex_cliques[v1]:
                            cover_count[ci] -= 1
                        for ci in vertex_cliques[v2]:
                            cover_count[ci] -= 1
                        current_set.discard(v1)
                        current_set.discard(v2)
                        current = list(current_set)
                        improved = True
                        found_swap = True
                        break
                    
                    # Find two vertices that cover all deficit cliques
                    # Try all pairs from the union of deficit clique vertices
                    deficit_vertices = set()
                    for ci in deficit_cliques:
                        deficit_vertices.update(cliques[ci])
                    deficit_vertices -= current_set
                    deficit_vertices.discard(v1)
                    deficit_vertices.discard(v2)
                    
                    dv_list = list(deficit_vertices)
                    if len(dv_list) <= 20:  # Limit search space
                        for a_idx in range(len(dv_list)):
                            if found_swap:
                                break
                            u1 = dv_list[a_idx]
                            for b_idx in range(a_idx + 1, len(dv_list)):
                                u2 = dv_list[b_idx]
                                # Check if {u1, u2} covers all deficit cliques
                                covers_all = True
                                for ci in deficit_cliques:
                                    if u1 not in cliques[ci] and u2 not in cliques[ci]:
                                        covers_all = False
                                        break
                                if covers_all:
                                    # Also check no new uncovered cliques
                                    new_uncovered = True
                                    for ci in set(vertex_cliques[u1]) | set(vertex_cliques[u2]):
                                        if cover_count[ci] == 0:
                                            new_uncovered = False
                                            break
                                    if not new_uncovered:
                                        continue
                                    # Apply swap
                                    for ci in vertex_cliques[v1]:
                                        cover_count[ci] -= 1
                                    for ci in vertex_cliques[v2]:
                                        cover_count[ci] -= 1
                                    for ci in vertex_cliques[u1]:
                                        cover_count[ci] += 1
                                    for ci in vertex_cliques[u2]:
                                        cover_count[ci] += 1
                                    current_set.discard(v1)
                                    current_set.discard(v2)
                                    current_set.add(u1)
                                    current_set.add(u2)
                                    current = list(current_set)
                                    improved = True
                                    found_swap = True
                                    break
            
            if found_swap:
                continue
    
    return sorted(current)

def _solve_instance(adj, cliques, time_budget=30.0):
    """Solve one instance: greedy + local search + B&B polish + tight polish."""
    if not cliques:
        return []
    
    start = time.time()
    
    # Step 1: Greedy
    greedy_sol = _greedy_set_cover(cliques)
    if not _validate_solution(greedy_sol, cliques):
        touched = set()
        for clique in cliques:
            touched.update(clique)
        greedy_sol = sorted(touched)
    
    best = greedy_sol
    best_size = len(greedy_sol)
    
    # Step 2: Local search on greedy
    remaining = time_budget - (time.time() - start)
    if remaining > 1:
        ls_sol = _local_search(adj, list(greedy_sol), cliques, max_iters=5000, time_budget=min(remaining, 10))
        if _validate_solution(ls_sol, cliques) and len(ls_sol) < best_size:
            best = ls_sol
            best_size = len(ls_sol)
    
    # Step 3: Random restarts
    remaining = time_budget - (time.time() - start)
    if remaining > 2:
        rs_sol = _random_local_search(adj, cliques, time_budget=min(remaining, 15))
        if _validate_solution(rs_sol, cliques) and len(rs_sol) < best_size:
            best = rs_sol
            best_size = len(rs_sol)
    
    # Step 4: B&B polish - try to find a solution strictly smaller than best
    remaining = time_budget - (time.time() - start)
    if remaining > 3 and best_size > 1:
        bb_sol = _branch_and_bound(cliques, best_size, time_budget=min(remaining, 15))
        if bb_sol is not None and _validate_solution(bb_sol, cliques) and len(bb_sol) < best_size:
            best = bb_sol
            best_size = len(bb_sol)
    
    # Step 5: Tight local search with 2-swaps for final polish
    remaining = time_budget - (time.time() - start)
    if remaining > 1:
        final_sol = _tight_local_search(adj, list(best), cliques, time_budget=min(remaining, 8))
        if _validate_solution(final_sol, cliques) and len(final_sol) < best_size:
            best = final_sol
            best_size = len(final_sol)
    
    # Step 6: One more B&B pass if time allows
    remaining = time_budget - (time.time() - start)
    if remaining > 2 and best_size > 1:
        bb_sol2 = _branch_and_bound(cliques, best_size, time_budget=min(remaining, 5))
        if bb_sol2 is not None and _validate_solution(bb_sol2, cliques) and len(bb_sol2) < best_size:
            best = bb_sol2
    
    return best

def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate time: give more time to larger/harder instances
    # Larger n with lower p tends to have more cliques, so weight by n^2 * p
    total_time = 135  # leave 10s buffer
    weights = [30**2 * 0.3, 35**2 * 0.25, 40**2 * 0.2]
    total_weight = sum(weights)
    time_alloc = [total_time * w / total_weight for w in weights]
    
    for i, (n, p, seed) in enumerate(instances):
        remaining_total = deadline - time.time()
        if remaining_total < 5:
            # Emergency fallback
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            sol = _greedy_set_cover(cliques)
            results.append(sol)
            continue
        
        # Reserve time for remaining instances
        remaining_instances = len(instances) - i - 1
        reserved = 5 * remaining_instances  # minimum 5s per remaining instance
        budget = min(time_alloc[i], remaining_total - reserved)
        if budget < 5:
            budget = max(3, (remaining_total - reserved) / 2)
        if budget < 3:
            budget = 3
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
        
        if not cliques:
            results.append([])
            continue
        
        sol = _solve_instance(adj, cliques, time_budget=budget)
        results.append(sol)
    
    return results

# EVOLVE-BLOCK-END
