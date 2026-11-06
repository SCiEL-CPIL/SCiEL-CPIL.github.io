import time
import random
import itertools
from collections import deque

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

def _greedy_lower_bound(uncovered_mask, clique_mask, n, vertex_cliques):
    """Compute a greedy lower bound on the number of additional vertices needed."""
    if uncovered_mask == 0:
        return 0
    
    remaining = uncovered_mask
    count = 0
    while remaining:
        # Find vertex covering most uncovered cliques
        best_v = -1
        best_count = 0
        for v in range(n):
            cnt = bin(clique_mask[v] & remaining).count('1')
            if cnt > best_count:
                best_count = cnt
                best_v = v
        if best_v == -1 or best_count == 0:
            break
        count += 1
        remaining &= ~clique_mask[best_v]
    return count

def _local_search_polish(solution, clique_mask, num_cliques, n, vertex_cliques):
    """Try to improve the solution by swapping vertices."""
    if not solution:
        return solution
    
    sol_set = set(solution)
    best_size = len(sol_set)
    
    # Try removing vertices
    for v in list(sol_set):
        test_set = sol_set - {v}
        covered = 0
        for cv in test_set:
            covered |= clique_mask[cv]
        if covered == (1 << num_cliques) - 1:
            sol_set.remove(v)
            best_size = len(sol_set)
    
    # Try swapping: remove one vertex, add another
    for v_remove in list(sol_set):
        without_v = sol_set - {v_remove}
        covered_without = 0
        for cv in without_v:
            covered_without |= clique_mask[cv]
        
        # Find uncovered cliques
        uncovered = ((1 << num_cliques) - 1) & ~covered_without
        if uncovered == 0:
            sol_set = without_v
            best_size = len(sol_set)
            continue
        
        # Try adding a vertex that covers uncovered cliques
        for v_add in range(n):
            if v_add in sol_set:
                continue
            if clique_mask[v_add] & uncovered:
                test_set = without_v | {v_add}
                if len(test_set) < best_size:
                    covered = 0
                    for cv in test_set:
                        covered |= clique_mask[cv]
                    if covered == (1 << num_cliques) - 1:
                        sol_set = test_set
                        best_size = len(sol_set)
                        break
    
    return sorted(sol_set)

def _solve_instance(adj, time_limit=40.0):
    """Solve the minimum transversal problem for maximal cliques using Branch and Bound."""
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []
    
    num_cliques = len(cliques)
    
    # Precompute vertex -> clique indices
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Bitmask representation
    # clique_mask[v] has bit i set if v is in clique i
    clique_mask = [0] * n
    for v in range(n):
        mask = 0
        for i in vertex_cliques[v]:
            mask |= (1 << i)
        clique_mask[v] = mask
        
    # Sort vertices by coverage (number of cliques) descending to help B&B ordering
    vertex_freq = [len(vertex_cliques[v]) for v in range(n)]
    
    # Precompute for each clique: list of vertices in it, sorted by frequency descending
    clique_vertices = []
    for i, clique in enumerate(cliques):
        verts = sorted(clique, key=lambda v: vertex_freq[v], reverse=True)
        clique_vertices.append(verts)
        
    best_solution = None
    best_size = num_cliques + 1 # Upper bound
    
    # Initial greedy solution to set a good upper bound
    S_greedy = set()
    uncovered_greedy = (1 << num_cliques) - 1
    while uncovered_greedy:
        # Find vertex covering most uncovered cliques
        best_v = -1
        best_count = -1
        for v in range(n):
            if v in S_greedy:
                continue
            # Count bits in (clique_mask[v] & uncovered_greedy)
            count = bin(clique_mask[v] & uncovered_greedy).count('1')
            if count > best_count:
                best_count = count
                best_v = v
        if best_v == -1 or best_count == 0:
            break
        S_greedy.add(best_v)
        uncovered_greedy &= ~clique_mask[best_v]
        
    if uncovered_greedy == 0:
        best_size = len(S_greedy)
        best_solution = set(S_greedy)
        
    # Post-optimization: try removing vertices from greedy solution
    if best_solution is not None:
        for v in list(best_solution):
            # Check if removing v still covers all cliques
            test_set = best_solution - {v}
            covered = 0
            for cv in test_set:
                covered |= clique_mask[cv]
            if covered == (1 << num_cliques) - 1:
                best_solution.remove(v)
                best_size = len(best_solution)
        
    # B&B Search
    start_time = time.time()
    timeout_flag = [False]
    
    def dfs(current_size, uncovered_mask, path):
        nonlocal best_size, best_solution
        
        if timeout_flag[0]:
            return False # Timeout
        
        if time.time() - start_time > time_limit:
            timeout_flag[0] = True
            return False # Timeout
        
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_solution = set(path)
            return True # Success
            
        # Pruning
        if current_size >= best_size:
            return False
            
        # Better lower bound: greedy lower bound on remaining
        lb = _greedy_lower_bound(uncovered_mask, clique_mask, n, vertex_cliques)
        if current_size + lb >= best_size:
            return False
        
        # Pick a clique to branch on: most constrained (fewest available vertices)
        # Find the uncovered clique with the fewest vertices that can still be chosen
        target_clique_idx = -1
        min_available = n + 1
        # Iterate through set bits in uncovered_mask
        um = uncovered_mask
        while um:
            idx = (um & -um).bit_length() - 1
            um &= um - 1
            # Count how many vertices in this clique are not yet used
            # (all vertices in clique_vertices[idx] are candidates since we only add to path)
            # Actually, all vertices in the clique are candidates for covering it
            avail = len(clique_vertices[idx])
            if avail < min_available:
                min_available = avail
                target_clique_idx = idx
        
        # Get vertices in this clique, sorted by coverage of uncovered cliques
        verts = clique_vertices[target_clique_idx]
        # Sort by how many uncovered cliques each vertex covers (descending)
        verts_sorted = sorted(verts, key=lambda v: bin(clique_mask[v] & uncovered_mask).count('1'), reverse=True)
        
        # Try each vertex in the clique
        for v in verts_sorted:
            new_uncovered = uncovered_mask & ~clique_mask[v]
            
            path.append(v)
            dfs(current_size + 1, new_uncovered, path)
            path.pop()
            
            if timeout_flag[0]:
                return False
            
            # If we found a solution with size <= current_size, we can prune
            if best_size <= current_size:
                return False
                
    dfs(0, (1 << num_cliques) - 1, [])
    
    if best_solution is not None:
        # Apply local search polish
        polished = _local_search_polish(sorted(best_solution), clique_mask, num_cliques, n, vertex_cliques)
        return polished
    else:
        # Fallback to greedy if B&B timed out without finding anything better
        if best_solution is None:
            return sorted(S_greedy)
        return sorted(best_solution)

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate more time to harder instances
    time_weights = [0.25, 0.35, 0.40]
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time() - 3  # 3s safety margin
        if remaining_time < 5:
            # Fallback: simple greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            # Simple greedy
            S = set()
            is_covered = [False] * len(cliques)
            vertex_cliques = [[] for _ in range(n)]
            for i, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].append(i)
                    
            remaining = list(range(len(cliques)))
            while remaining:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if v in S: continue
                    cnt = sum(1 for ci in vertex_cliques[v] if not is_covered[ci])
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v == -1: break
                S.add(best_v)
                for ci in vertex_cliques[best_v]:
                    if not is_covered[ci]:
                        is_covered[ci] = True
                        remaining.remove(ci)
            results.append(sorted(S))
            continue
            
        time_budget = remaining_time * time_weights[idx]
        adj = _build_graph(n, p, seed)
        sol = _solve_instance(adj, time_limit=time_budget)
        results.append(sol)
        
    return results

# EVOLVE-BLOCK-END
