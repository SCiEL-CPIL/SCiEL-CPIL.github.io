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
    """Local search: try removing vertices if still valid, or swapping."""
    start = time.time()
    current = list(initial_solution)
    n = len(adj)
    
    if not cliques:
        return sorted(current)
    
    def is_valid(sol):
        sol_set = set(sol)
        for clique in cliques:
            if not (sol_set & clique):
                return False
        return True
    
    for _ in range(max_iters):
        if time.time() - start > time_budget:
            break
        improved = False
        
        # Try removing each vertex
        for i, v in enumerate(list(current)):
            trial = current[:i] + current[i+1:]
            if is_valid(trial):
                current = trial
                improved = True
                break
        if improved:
            continue
        
        # Try 1-out-1-in swaps: remove v, add u not in current
        all_vertices = set(range(n))
        for v in list(current):
            trial_without = set(current) - {v}
            # Only consider adding vertices that are in at least one clique
            candidate_adds = []
            for u in all_vertices - trial_without:
                if u not in current:
                    candidate_adds.append(u)
            
            for u in candidate_adds:
                trial = trial_without | {u}
                if len(trial) >= len(current):
                    continue
                if is_valid(trial):
                    current = list(trial)
                    improved = True
                    break
            if improved:
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
    
    # Random restarts: pick random subsets, then add needed vertices
    for _ in range(30):
        if time.time() - start > time_budget:
            break
        
        # Start with a random subset
        subset_size = rng.randint(0, min(n, best_size + 2 if best else 5))
        candidate = set(rng.sample(range(n), subset_size))
        
        # Greedily add vertices to cover all cliques
        covered_cliques = set()
        for i, clique in enumerate(cliques):
            if candidate & set(clique):
                covered_cliques.add(i)
        
        while len(covered_cliques) < len(cliques):
            # Find uncovered cliques
            uncovered = [cliques[i] for i in range(len(cliques)) if i not in covered_cliques]
            if not uncovered:
                break
            # Pick vertex that covers most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in candidate:
                    count = sum(1 for c in uncovered if v in c)
                else:
                    count = sum(1 for c in uncovered if v in c)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            candidate.add(best_v)
            for i, clique in enumerate(cliques):
                if i not in covered_cliques and best_v in clique:
                    covered_cliques.add(i)
        
        if len(covered_cliques) < len(cliques):
            continue
        
        if len(candidate) < best_size:
            # Local search to improve
            improved = _local_search(adj, list(candidate), cliques, max_iters=200, time_budget=1.0)
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
            better = _random_local_search(adj, cliques, time_budget=min(remaining, 20))
            if _validate_solution(adj, better, cliques) and len(better) < len(greedy_sol):
                greedy_sol = better
        
        results.append(greedy_sol)
    
    return results

# EVOLVE-BLOCK-END
