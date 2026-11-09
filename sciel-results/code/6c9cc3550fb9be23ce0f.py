import time
import sys
import math
from random import Random
from collections import defaultdict

def _random_graph_edges(n, p, seed):
    """Generate edges of a random graph using the same procedure as the seed program."""
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

def _build_adjacency(n, edges):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch_maximal_cliques(n, adj):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        PX = P | X
        if not PX:
            return
        best_pivot = None
        best_count = -1
        for u in PX:
            count = len(P & adj[u])
            if count > best_count:
                best_count = count
                best_pivot = u
        candidates = list(P - adj[best_pivot])
        for v in candidates:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(frozenset(), set(range(n)), set())
    return cliques

def _verify_hitting_set(selected, cliques):
    """Verify that selected vertices hit every clique."""
    selected_set = set(selected)
    for clique in cliques:
        if not (clique & selected_set):
            return False
    return True

def _greedy_hitting_set(n, cliques):
    """Greedy heuristic for minimum hitting set using bitsets for speed."""
    if not cliques:
        return []
    
    m = len(cliques)
    # Map vertex to bitmask of cliques it covers
    vertex_masks = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_masks[v] |= bit
    
    uncovered = (1 << m) - 1
    selected = []
    remaining_cliques = m
    
    while uncovered:
        best_v = -1
        best_count = 0
        # Find vertex covering most uncovered cliques
        for v in range(n):
            mask = vertex_masks[v] & uncovered
            count = mask.bit_count()
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v == -1:
            break
            
        selected.append(best_v)
        uncovered &= ~vertex_masks[best_v]
        remaining_cliques -= best_count # Approximation, but we just check uncovered != 0
        
    return selected

def _solve_bnb_bitset(n, cliques, initial_solution, time_limit=10.0):
    """Solve hitting set using Branch and Bound with bitsets."""
    if not cliques:
        return []
    
    m = len(cliques)
    start_time = time.time()
    
    # Precompute masks
    vertex_masks = [0] * n
    clique_vertices = [set() for _ in range(m)]
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_masks[v] |= bit
            clique_vertices[i].add(v)
    
    initial_mask = (1 << m) - 1
    best_size = len(initial_solution)
    best_solution = list(initial_solution)
    
    # Order vertices by coverage (descending) to find good solutions early?
    # Or just use the branching strategy.
    
    # Precompute for each clique, the list of vertices in it
    # We need to pick a clique to branch on.
    
    def dfs(uncovered_mask, depth, current_vertices):
        nonlocal best_size, best_solution, start_time
        
        if time.time() - start_time > time_limit:
            return False # Timeout signal
        
        if uncovered_mask == 0:
            if depth < best_size:
                best_size = depth
                best_solution = list(current_vertices)
            return True
        
        # Lower bound: We need at least 1 more vertex.
        # Better: Max coverage of any single vertex is C.
        # ceil(popcount(uncovered) / C) is a lower bound on additional vertices.
        popcount_uncovered = uncovered_mask.bit_count()
        
        max_cov = 0
        for v in range(n):
            cov = (vertex_masks[v] & uncovered_mask).bit_count()
            if cov > max_cov:
                max_cov = cov
        
        if max_cov == 0:
            return False # Should not happen if uncovered != 0 and graph is connected?
            
        lb = math.ceil(popcount_uncovered / max_cov)
        if depth + lb >= best_size:
            return True # Pruned, but not timeout
        
        # Select a clique to branch on: pick one with fewest candidate vertices
        # Candidates for clique i are vertices v in clique i such that vertex_masks[v] & uncovered_mask != 0
        # Actually, any vertex in the clique covers the clique.
        # We want to pick a clique that restricts the search.
        # Heuristic: pick clique with minimum number of vertices that cover *any* uncovered clique?
        # Simpler: pick the first uncovered clique.
        
        # Find index of first set bit in uncovered_mask
        # Since m can be > 64, we can't use simple bit ops for index easily in all Python versions?
        # Python int.bit_length works.
        # To find lowest set bit:
        low_bit = uncovered_mask & -uncovered_mask
        clique_idx = low_bit.bit_length() - 1
        
        # Get vertices in this clique
        candidates = list(clique_vertices[clique_idx])
        
        # Sort candidates by coverage of uncovered cliques (descending) to find good solutions faster
        candidates.sort(key=lambda v: (vertex_masks[v] & uncovered_mask).bit_count(), reverse=True)
        
        for v in candidates:
            new_uncovered = uncovered_mask & ~vertex_masks[v]
            if dfs(new_uncovered, depth + 1, current_vertices + [v]):
                return True # Timeout propagated
            
    # Start DFS
    # We can also try to improve initial solution with local search first to tighten bound
    # But let's just run B&B.
    
    # To make B&B effective, we need a good initial bound.
    # We have initial_solution from greedy.
    
    dfs(initial_mask, 0, [])
    
    return best_solution

def _local_search_improve(selected, cliques, n, time_limit=20.0):
    """Improve the hitting set via local search."""
    start_time = time.time()
    selected_set = set(selected)
    clique_sets = [set(c) for c in cliques]
    m = len(clique_sets)
    
    # Precompute vertex -> cliques
    vertex_cliques = defaultdict(list)
    for i, c in enumerate(clique_sets):
        for v in c:
            vertex_cliques[v].append(i)
    
    # Coverage count for each clique
    coverage = [0] * m
    for v in selected_set:
        for ci in vertex_cliques[v]:
            coverage[ci] += 1
    
    def remove_redundant():
        """Remove all redundant vertices."""
        changed = True
        while changed and (time.time() - start_time) < time_limit:
            changed = False
            for v in list(selected_set):
                if (time.time() - start_time) > time_limit:
                    return
                can_remove = True
                for ci in vertex_cliques[v]:
                    if coverage[ci] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ci in vertex_cliques[v]:
                        coverage[ci] -= 1
                    selected_set.discard(v)
                    changed = True
    
    # Phase 1: Remove redundant vertices
    remove_redundant()
    
    # Phase 2: Deterministic local search (1-swap, 2-for-1)
    # Try 1-swap: remove one, add one
    for _ in range(300):
        if (time.time() - start_time) > time_limit:
            break
        improved = False
        for v in list(selected_set):
            if (time.time() - start_time) > time_limit:
                break
            # Remove v
            for ci in vertex_cliques[v]:
                coverage[ci] -= 1
            selected_set.discard(v)
            
            uncovered = [ci for ci in range(m) if coverage[ci] == 0]
            if not uncovered:
                improved = True
                break
            
            common = set(clique_sets[uncovered[0]])
            for ci in uncovered[1:]:
                common &= clique_sets[ci]
            
            for w in common:
                if w not in selected_set:
                    selected_set.add(w)
                    for ci in vertex_cliques[w]:
                        coverage[ci] += 1
                    improved = True
                    break
            
            if improved:
                break
            
            selected_set.add(v)
            for ci in vertex_cliques[v]:
                coverage[ci] += 1
        
        if not improved:
            break
    
    # Try 2-for-1: remove two, add one
    for _ in range(100):
        if (time.time() - start_time) > time_limit:
            break
        improved = False
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, len(sel_list)):
                if (time.time() - start_time) > time_limit:
                    break
                v1, v2 = sel_list[i], sel_list[j]
                for ci in vertex_cliques[v1]:
                    coverage[ci] -= 1
                for ci in vertex_cliques[v2]:
                    coverage[ci] -= 1
                selected_set.discard(v1)
                selected_set.discard(v2)
                
                uncovered = [ci for ci in range(m) if coverage[ci] == 0]
                if not uncovered:
                    improved = True
                    break
                
                common = set(clique_sets[uncovered[0]])
                for ci in uncovered[1:]:
                    common &= clique_sets[ci]
                
                for w in common:
                    if w not in selected_set:
                        selected_set.add(w)
                        for ci in vertex_cliques[w]:
                            coverage[ci] += 1
                        improved = True
                        break
                
                if improved:
                    break
                
                selected_set.add(v1)
                selected_set.add(v2)
                for ci in vertex_cliques[v1]:
                    coverage[ci] += 1
                for ci in vertex_cliques[v2]:
                    coverage[ci] += 1
            
            if improved:
                break
        
        if not improved:
            break
    
    remove_redundant()
    
    return sorted(selected_set)

def _solve_hitting_set(n, cliques, time_budget=40.0):
    """Solve the minimum hitting set problem with multiple strategies."""
    if not cliques:
        return []
    
    start_time = time.time()
    
    # Get a good initial solution with greedy
    greedy_sol = _greedy_hitting_set(n, cliques)
    best_sol = greedy_sol
    best_cost = len(best_sol)
    
    # Local search improvement
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 2.0:
        improved = _local_search_improve(best_sol, cliques, n, time_limit=remaining_time * 0.5)
        if len(improved) < best_cost:
            best_sol = improved
            best_cost = len(best_sol)
    
    # Try B&B with the improved solution as upper bound
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 1.0:
        bnb_sol = _solve_bnb_bitset(n, cliques, best_sol, time_limit=remaining_time)
        if len(bnb_sol) < best_cost:
            best_sol = bnb_sol
            best_cost = len(best_sol)
    
    return best_sol

def construct():
    """Main entry point. Returns three vertex lists, one per instance."""
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        # Check time
        if time.time() > deadline:
            edges = _random_graph_edges(n, p, seed)
            touched = {x for e in edges for x in e}
            out.append(sorted(touched))
            continue
        
        edges = _random_graph_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        cliques = _bron_kerbosch_maximal_cliques(n, adj)
        
        if not cliques:
            out.append([])
            continue
        
        # Allocate time budget for this instance
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        # Give more time to later instances since they may be harder
        if n == 30:
            instance_budget = min(35.0, remaining * 0.7)
        elif n == 35:
            instance_budget = min(40.0, remaining * 0.6)
        else:
            instance_budget = min(45.0, remaining * 0.5)
        
        # Solve hitting set
        selected = _solve_hitting_set(n, cliques, time_budget=instance_budget)
        
        # Final verification
        if not _verify_hitting_set(selected, cliques):
            touched = {x for e in edges for x in e}
            selected = sorted(touched)
        
        out.append(sorted(selected))
    
    return out
