"""Combined program: Bron-Kerbosch clique enumeration + greedy set cover + aggressive local search."""
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
        # Choose pivot u from P union X to reduce recursion
        U = P | X
        if not U:
            return
        # Pick pivot with max degree in P (heuristic)
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

def _validate_solution(adj, selected, cliques):
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
    
    n = max(max(c) for c in cliques) + 1
    # Build vertex -> cliques mapping
    vertex_to_cliques = {}
    for i, clique in enumerate(cliques):
        for v in clique:
            if v not in vertex_to_cliques:
                vertex_to_cliques[v] = []
            vertex_to_cliques[v].append(i)
    
    covered = [False] * len(cliques)
    selected = []
    
    while any(not c for c in covered):
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
            covered[idx] = True
    
    return sorted(selected)

def _local_search(adj, initial_solution, cliques, max_iters=2000, time_budget=5.0):
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
    
    # Helper to check if a clique is covered
    def is_covered(ci):
        return cover_count[ci] > 0
    
    for _ in range(max_iters):
        if time.time() - start > time_budget:
            break
        improved = False
        current_set = set(current)
        
        # 1. Try simple removals
        for v in list(current):
            # Check if removing v keeps all its cliques covered
            safe_to_remove = True
            for ci in vertex_cliques[v]:
                if cover_count[ci] == 1:
                    safe_to_remove = False
                    break
            if safe_to_remove:
                # Remove v
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
            # Tentatively remove v
            deficit_cliques = []
            for ci in vertex_cliques[v]:
                if cover_count[ci] == 1:
                    deficit_cliques.append(ci)
            
            if not deficit_cliques:
                # This case should have been caught by removal, but just in case
                for ci in vertex_cliques[v]:
                    cover_count[ci] -= 1
                current_set.remove(v)
                current = list(current_set)
                improved = True
                break
            
            # Find a vertex u not in current_set that covers ALL deficit_cliques
            # Optimization: If only 1 deficit clique, pick any u in that clique not in set
            # If multiple, intersect sets
            if len(deficit_cliques) == 1:
                ci = deficit_cliques[0]
                candidates = [u for u in cliques[ci] if u not in current_set]
                if candidates:
                    u = candidates[0]
                    # Apply swap
                    for c1 in vertex_cliques[v]:
                        cover_count[c1] -= 1
                    for c2 in vertex_cliques[u]:
                        cover_count[c2] += 1
                    current_set.remove(v)
                    current_set.add(u)
                    current = list(current_set)
                    improved = True
                    break
            else:
                # Intersect vertices from deficit cliques
                # Start with vertices of the first deficit clique
                potential_u = set(cliques[deficit_cliques[0]])
                for ci in deficit_cliques[1:]:
                    potential_u &= set(cliques[ci])
                
                # Filter out current_set
                potential_u -= current_set
                if potential_u:
                    u = next(iter(potential_u))
                    # Apply swap
                    for c1 in vertex_cliques[v]:
                        cover_count[c1] -= 1
                    for c2 in vertex_cliques[u]:
                        cover_count[c2] += 1
                    current_set.remove(v)
                    current_set.add(u)
                    current = list(current_set)
                    improved = True
                    break
        
        if not improved:
            break
    
    return sorted(current)

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
    if _validate_solution(adj, greedy_sol, cliques):
        best = greedy_sol
        best_size = len(greedy_sol)
    
    # Precompute vertex -> clique indices for fast coverage checks
    n_cliques = len(cliques)
    vertex_cliques = [set() for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].add(i)
    
    # Random restarts: pick random subsets, then add needed vertices
    # Increase restarts since local search is faster
    max_restarts = 100 
    for r in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        # Start with a random subset
        # Bias towards smaller subsets to find better solutions
        max_subset = min(n, best_size + 2 if best else 5)
        subset_size = rng.randint(0, max_subset)
        candidate = set(rng.sample(range(n), subset_size))
        
        # Greedily add vertices to cover all cliques
        covered_cliques = set()
        for i, clique in enumerate(cliques):
            if candidate & clique:
                covered_cliques.add(i)
        
        while len(covered_cliques) < n_cliques:
            # Find uncovered cliques
            uncovered = [i for i in range(n_cliques) if i not in covered_cliques]
            if not uncovered:
                break
            # Pick vertex that covers most uncovered cliques
            best_v = None
            best_count = -1
            # Only check vertices that are in uncovered cliques to save time
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
            # Local search to improve
            # Use a smaller time budget per restart to allow more restarts
            improved = _local_search(adj, list(candidate), cliques, max_iters=1000, time_budget=0.5)
            if _validate_solution(adj, improved, cliques) and len(improved) < best_size:
                best = improved
                best_size = len(improved)
    
    if best is None:
        return greedy_sol
    return best

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline - 10:
            # Fallback: use a simple heuristic
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            sol = _greedy_set_cover(cliques)
            results.append(sol)
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
        
        if not cliques:
            results.append([])
            continue
        
        # Try greedy first
        greedy_sol = _greedy_set_cover(cliques)
        if not _validate_solution(adj, greedy_sol, cliques):
            # Should not happen, but fallback
            touched = set()
            for clique in cliques:
                touched.update(clique)
            greedy_sol = sorted(touched)
        
        # Try local search / random restarts with remaining time
        remaining = deadline - time.time() - 5
        if remaining > 1:
            # Allocate more time to random restarts as they explore different basins
            better = _random_local_search(adj, cliques, time_budget=min(remaining, 30))
            if _validate_solution(adj, better, cliques) and len(better) < len(greedy_sol):
                greedy_sol = better
            # Extra local search pass on the best solution found
            # Use remaining time for a final polish
            final_time = max(0, min(remaining, 15))
            final = _local_search(adj, list(greedy_sol), cliques, max_iters=5000, time_budget=final_time)
            if _validate_solution(adj, final, cliques) and len(final) < len(greedy_sol):
                greedy_sol = final
        
        results.append(greedy_sol)
    
    return results

# EVOLVE-BLOCK-END
