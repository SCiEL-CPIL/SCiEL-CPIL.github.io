import time
import random
import sys
from collections import defaultdict

def _build_graph(n, p, seed):
    """Build adjacency lists from the same PRNG stream as the seed program."""
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

def _greedy_hitting_set(vertex_covers, num_cliques):
    """Greedy set cover to get an initial hitting set."""
    n = len(vertex_covers)
    sol_mask = 0
    remaining = (1 << num_cliques) - 1
    
    while remaining:
        best_v = -1
        best_count = 0
        for v in range(n):
            if (sol_mask & (1 << v)) != 0:
                continue
            count = (vertex_covers[v] & remaining).bit_count()
            if count > best_count:
                best_count = count
                best_v = v
        if best_v == -1 or best_count == 0:
            break
        sol_mask |= (1 << best_v)
        remaining &= ~vertex_covers[best_v]
    
    return sol_mask

def _packing_lower_bound(clique_vertices, remaining_mask, num_cliques):
    """
    Compute a lower bound via maximum matching on the bipartite graph
    between remaining cliques and vertices. If we can find k disjoint cliques
    (each needing a distinct vertex), we need at least k more vertices.
    Uses greedy matching as approximation.
    """
    if remaining_mask == 0:
        return 0
    
    # Greedy matching: repeatedly pick the clique with fewest available vertices
    # and match it to the vertex that covers the fewest other remaining cliques
    matched_cliques = set()
    matched_vertices = set()
    count = 0
    
    # Get list of remaining cliques
    rem_cliques = []
    c = remaining_mask
    while c:
        i = (c & -c).bit_length() - 1
        c &= c - 1
        rem_cliques.append(i)
    
    # Sort by number of available vertices (fewest first)
    rem_cliques.sort(key=lambda i: len(clique_vertices[i]))
    
    for ci in rem_cliques:
        if ci in matched_cliques:
            continue
        # Find available vertices for this clique
        avail = [v for v in clique_vertices[ci] if v not in matched_vertices]
        if not avail:
            continue
        # Pick vertex that covers fewest other remaining cliques (most specific)
        best_v = min(avail, key=lambda v: sum(1 for cj in rem_cliques if cj != ci and cj not in matched_cliques and v in clique_vertices[cj]))
        matched_vertices.add(best_v)
        matched_cliques.add(ci)
        count += 1
    
    return count

def _solve_instance(n, p, seed, time_budget):
    """
    Solve the minimum hitting set problem for maximal cliques of a random graph.
    Uses branch-and-bound with constraint propagation.
    """
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []

    num_cliques = len(cliques)
    
    # Precompute for each vertex, the bitmask of cliques it covers
    vertex_covers = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_covers[v] |= bit
    
    # Remove vertices that don't cover any clique
    active_vertices = [v for v in range(n) if vertex_covers[v] != 0]
    if not active_vertices:
        return []
    
    # Remap to compact indices
    v_to_compact = {v: i for i, v in enumerate(active_vertices)}
    compact_n = len(active_vertices)
    compact_covers = [0] * compact_n
    for v in active_vertices:
        compact_covers[v_to_compact[v]] = vertex_covers[v]
    
    # Get initial upper bound from greedy
    greedy_sol = _greedy_hitting_set(compact_covers, num_cliques)
    greedy_size = greedy_sol.bit_count()
    
    # Apply local search to improve the greedy solution
    ls_sol = _local_search_improve(
        [i for i in range(compact_n) if (greedy_sol & (1 << i))],
        cliques, compact_covers, num_cliques, compact_n
    )
    ls_mask = 0
    for v in ls_sol:
        ls_mask |= (1 << v)
    ls_size = ls_mask.bit_count()
    
    if ls_size < greedy_size:
        greedy_sol = ls_mask
        greedy_size = ls_size
    
    # If greedy is very small, try to prove optimality or improve
    if greedy_size <= 1:
        return [active_vertices[i] for i in range(compact_n) if (greedy_sol & (1 << i))]
    
    # Precompute: for each clique, list of vertices that cover it
    clique_vertices = [[] for _ in range(num_cliques)]
    for v in range(compact_n):
        c = compact_covers[v]
        while c:
            i = (c & -c).bit_length() - 1
            c &= c - 1
            clique_vertices[i].append(v)
    
    # Sort cliques by size (fewest vertices first) for better pruning
    clique_order = sorted(range(num_cliques), key=lambda i: len(clique_vertices[i]))
    
    best_size = [greedy_size]
    best_sol = [greedy_sol]
    start_time = time.time()
    deadline = start_time + time_budget
    
    def bb(selected_mask, remaining_cliques_mask):
        """
        Branch and bound for minimum hitting set.
        selected_mask: bitmask of vertices already selected
        remaining_cliques_mask: bitmask of cliques not yet hit
        """
        if time.time() > deadline:
            return
        
        if remaining_cliques_mask == 0:
            size = selected_mask.bit_count()
            if size < best_size[0]:
                best_size[0] = size
                best_sol[0] = selected_mask
            return
        
        # Lower bound using packing (maximum matching approximation)
        current_size = selected_mask.bit_count()
        
        # Quick check: at least 1 more needed
        if current_size + 1 >= best_size[0]:
            return
        
        # Packing lower bound: find disjoint remaining cliques
        lb = _packing_lower_bound(clique_vertices, remaining_cliques_mask, num_cliques)
        if current_size + lb >= best_size[0]:
            return
        
        # Find the remaining clique with fewest candidate vertices
        best_clique = -1
        best_count = float('inf')
        
        c = remaining_cliques_mask
        while c:
            i = (c & -c).bit_length() - 1
            c &= c - 1
            # Count vertices in clique i not yet selected
            count = 0
            for v in clique_vertices[i]:
                if (selected_mask & (1 << v)) == 0:
                    count += 1
            if count < best_count:
                best_count = count
                best_clique = i
                if count == 1:
                    break
        
        if best_clique == -1:
            return
        
        # If any remaining clique has 0 available vertices, this branch is infeasible
        if best_count == 0:
            return
        
        # Branch on each vertex in the most constrained clique
        candidates = []
        for v in clique_vertices[best_clique]:
            if (selected_mask & (1 << v)) == 0:
                candidates.append(v)
        
        # Sort candidates by how many remaining cliques they cover (most first)
        candidates.sort(key=lambda v: (compact_covers[v] & remaining_cliques_mask).bit_count(), reverse=True)
        
        for v in candidates:
            if time.time() > deadline:
                return
            
            new_selected = selected_mask | (1 << v)
            new_remaining = remaining_cliques_mask & ~compact_covers[v]
            
            # Prune: if we've already selected as many as best, no point continuing
            if new_selected.bit_count() >= best_size[0]:
                continue
            
            bb(new_selected, new_remaining)
    
    # Run branch and bound
    bb(0, (1 << num_cliques) - 1)
    
    # Map back to original vertex indices
    result = []
    for i in range(compact_n):
        if (best_sol[0] & (1 << i)) != 0:
            result.append(active_vertices[i])
    
    # Final local search polish on the best solution found
    if time.time() < deadline - 2:
        final_ls = _local_search_improve(result, cliques, compact_covers, num_cliques, compact_n)
        if len(final_ls) < len(result):
            result = final_ls
    
    return sorted(result)

def _local_search_improve(sol_set, cliques, compact_covers, num_cliques, n_vertices, max_iter=20000):
    """
    Improved local search: iterative removal + 1-swap that enables further removals.
    Works on compact indices with bitmask clique covers.
    """
    sol_mask = 0
    for v in sol_set:
        sol_mask |= (1 << v)
    
    all_cliques_mask = (1 << num_cliques) - 1
    
    # Phase 1: Remove redundant vertices (those whose removal still hits all cliques)
    changed = True
    while changed:
        changed = False
        for v in range(n_vertices):
            if (sol_mask & (1 << v)) == 0:
                continue
            test_mask = sol_mask & ~(1 << v)
            # Check if all cliques are still hit
            # A clique i is hit if any selected vertex covers it
            # Equivalent: for every clique bit set in all_cliques_mask, 
            # the union of covers of selected vertices covers it
            # Faster: check that remaining cliques after removing v are still covered
            # A clique is uncovered if no selected vertex has that clique bit
            # Let's compute: which cliques are covered by test_mask?
            covered = 0
            for u in range(n_vertices):
                if (test_mask & (1 << u)):
                    covered |= compact_covers[u]
            if (covered & all_cliques_mask) == all_cliques_mask:
                sol_mask = test_mask
                changed = True
    
    # Phase 2: Try 1-exchange (remove v_out, add v_in) if it enables net removal
    for _ in range(max_iter):
        improved = False
        for v_out in range(n_vertices):
            if (sol_mask & (1 << v_out)) == 0:
                continue
            test_mask = sol_mask & ~(1 << v_out)
            
            # Find uncovered cliques
            covered = 0
            for u in range(n_vertices):
                if (test_mask & (1 << u)):
                    covered |= compact_covers[u]
            uncovered = all_cliques_mask & ~covered
            
            if uncovered == 0:
                # v_out is redundant (shouldn't happen after Phase 1, but just in case)
                sol_mask = test_mask
                improved = True
                break
            
            # Try to find a single vertex that covers all uncovered cliques
            # AND doesn't create redundancy issues
            for v_in in range(n_vertices):
                if (test_mask & (1 << v_in)):
                    continue
                if (compact_covers[v_in] & uncovered) != uncovered:
                    continue
                # This swap: remove v_out, add v_in. Same size.
                # But it might enable further removals. Try it.
                new_mask = test_mask | (1 << v_in)
                
                # Now try to remove redundant vertices from new_mask
                new_mask2 = new_mask
                for u in range(n_vertices):
                    if (new_mask2 & (1 << u)) == 0:
                        continue
                    tm = new_mask2 & ~(1 << u)
                    cov2 = 0
                    for w in range(n_vertices):
                        if (tm & (1 << w)):
                            cov2 |= compact_covers[w]
                    if (cov2 & all_cliques_mask) == all_cliques_mask:
                        new_mask2 = tm
                
                # If we got smaller, accept
                if new_mask2.bit_count() < sol_mask.bit_count():
                    sol_mask = new_mask2
                    improved = True
                    break
        
        if improved:
            continue
        else:
            break
    
    return [v for v in range(n_vertices) if (sol_mask & (1 << v))]

def construct():
    deadline = time.time() + 145  # Safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for i, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            # Fallback: just return greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            # Simple greedy
            sol = []
            for clique in cliques:
                if not any(v in sol for v in clique):
                    v = next(iter(clique))
                    sol.append(v)
            results.append(sorted(set(sol)))
            continue
         
        # Allocate time: give more to earlier instances, but ensure all get some
        if i == 0:
            alloc = min(remaining_time - 10, 50)
        elif i == 1:
            alloc = min(remaining_time - 5, 40)
        else:
            alloc = min(remaining_time - 2, 30)
        
        sol = _solve_instance(n, p, seed, alloc)
        results.append(sol)
        
    return results
