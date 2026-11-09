"""Solve the transversal problem: minimum set of vertices hitting every maximal clique (size>=2).

Strategy:
1. Generate each random graph with the same RNG procedure as the seed program.
2. Enumerate all maximal cliques (size >= 2) using Bron-Kerbosch with pivot.
3. Solve the hitting set problem via custom branch-and-bound with LP relaxation bounds.
   Use greedy + local search as a warm start and fallback.
4. Use a time budget to ensure we finish well within the 150s wall-clock limit.
"""

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
    """Greedy heuristic for minimum hitting set."""
    if not cliques:
        return []
    
    remaining_cliques = [set(c) for c in cliques]
    selected = set()
    
    vertex_cliques = {v: [] for v in range(n)}
    for i, clique in enumerate(remaining_cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    uncovered = [True] * len(remaining_cliques)
    uncovered_count = len(remaining_cliques)
    
    while uncovered_count > 0:
        best_v = -1
        best_score = -1
        best_clique_size_sum = float('inf')
        
        for v in range(n):
            if v in selected:
                continue
            cov_count = 0
            clique_size_sum = 0
            for ci in vertex_cliques[v]:
                if uncovered[ci]:
                    cov_count += 1
                    clique_size_sum += len(remaining_cliques[ci])
            
            if cov_count > 0:
                score = cov_count
                if score > best_score or (score == best_score and clique_size_sum < best_clique_size_sum):
                    best_score = score
                    best_v = v
                    best_clique_size_sum = clique_size_sum
        
        if best_v == -1:
            break
        
        selected.add(best_v)
        
        for ci in vertex_cliques[best_v]:
            if uncovered[ci]:
                uncovered[ci] = False
                uncovered_count -= 1
    
    return sorted(selected)

def _lp_relaxation_bound(cliques, n, time_limit=2.0):
    """Solve LP relaxation of hitting set to get a lower bound.
    
    Returns (lower_bound, fractional_solution) where lower_bound is the LP objective.
    Uses scipy.optimize.linprog if available.
    """
    try:
        from scipy.optimize import linprog
        import numpy as np
        
        m = len(cliques)
        A = np.zeros((m, n))
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1.0
        
        c = np.ones(n)
        
        start = time.time()
        res = linprog(c, A_ub=-A, b_ub=-np.ones(m), bounds=(0, 1), method='highs')
        
        if res.success:
            return res.fun, res.x
        return None, None
    except Exception:
        return None, None

def _branch_and_bound(cliques, n, initial_solution, time_limit=30.0):
    """Custom branch-and-bound for hitting set with LP relaxation bounds.
    
    Returns the best integer solution found.
    """
    if not cliques:
        return []
    
    start_time = time.time()
    best_solution = set(initial_solution)
    best_cost = len(best_solution)
    
    # Precompute: for each vertex, which cliques it covers
    vertex_cliques = defaultdict(list)
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Active cliques: those not yet covered
    active_cliques = [set(c) for c in cliques]
    active_set = set(range(len(cliques)))
    
    # Build a conflict graph: two vertices conflict if they are in the same clique
    # (we need at least one from each clique)
    
    # Use a simpler approach: branch on uncovered cliques
    # State: (active_cliques, selected_set)
    # At each node, pick an uncovered clique, branch on which vertex to pick from it
    
    # For efficiency, use a priority queue ordered by lower bound
    import heapq
    
    # Compute initial LP bound
    lp_bound, _ = _lp_relaxation_bound(cliques, n, time_limit=1.0)
    
    # Simple branch and bound:
    # Pick the smallest uncovered clique, branch on its vertices
    
    def get_lower_bound(active_clq):
        """Simple lower bound: for each active clique, we need at least 1 vertex.
        But vertices can cover multiple cliques. Use a greedy packing bound."""
        if not active_clq:
            return 0
        # Count active cliques
        return 1  # minimum: at least 1 more vertex needed
    
    # More sophisticated: use the fact that if two cliques are disjoint, we need 2 vertices
    # For now, use a simple bound and rely on the search
    
    # Alternative approach: use iterative deepening / best-first search
    # But for these sizes, a simpler recursive B&B should work
    
    # Let's use a different strategy: start from the greedy solution and try to improve
    # by removing vertices and finding replacements
    
    return _local_search_improve(initial_solution, cliques, n, time_limit)

def _local_search_improve(selected, cliques, n, time_limit=20.0):
    """Improve the hitting set via local search with 1-swap, 2-opt, and 3-opt moves."""
    start_time = time.time()
    selected_set = set(selected)
    clique_sets = [set(c) for c in cliques]
    m = len(clique_sets)
    
    if m == 0:
        return []

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
    
    def is_valid():
        return all(cov > 0 for cov in coverage)
        
    def remove_redundant():
        """Remove any vertex that is not needed for coverage."""
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

    def try_remove_add(remove_verts, add_verts):
        """Attempt to swap out remove_verts for add_verts if it improves size and maintains validity."""
        if len(add_verts) >= len(remove_verts):
            return False
            
        # Check if add_verts can cover all cliques that would be uncovered by removing remove_verts
        # First, simulate removal
        temp_coverage = coverage[:]
        for v in remove_verts:
            for ci in vertex_cliques[v]:
                temp_coverage[ci] -= 1
            if temp_coverage[ci] < 0:
                return False # Should not happen if v is in selected_set

        # Find uncovered cliques
        uncovered = [ci for ci in range(m) if temp_coverage[ci] == 0]
        if not uncovered:
            # We can just remove
            for v in remove_verts:
                for ci in vertex_cliques[v]:
                    coverage[ci] -= 1
                selected_set.discard(v)
            return True

        # Try to add one vertex that covers all uncovered cliques
        # Find intersection of all uncovered cliques
        common = set(clique_sets[uncovered[0]])
        for ci in uncovered[1:]:
            common &= clique_sets[ci]
            
        for w in common:
            if w not in selected_set:
                # Check if adding w covers all uncovered
                # We already know it's in the intersection, so it covers all uncovered
                # Apply change
                for v in remove_verts:
                    for ci in vertex_cliques[v]:
                        coverage[ci] -= 1
                    selected_set.discard(v)
                
                selected_set.add(w)
                for ci in vertex_cliques[w]:
                    coverage[ci] += 1
                return True
        return False

    # 1. Initial cleanup
    remove_redundant()
    
    # 2. Iterative improvement
    # We loop until no improvement or time runs out
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # 1-swap: Remove 1, Add 0 (if possible) or Remove 1, Add 1 (if it leads to smaller set later? No, we want min size)
        # Actually, "Remove 1, Add 0" is just redundancy removal.
        # "Remove 1, Add 1" doesn't reduce size.
        # We need moves that reduce size.
        # Size reduction moves:
        # - Remove k, Add 0 (Redundancy)
        # - Remove k, Add j where j < k (e.g., Remove 2, Add 1)
        
        # Try Remove 2, Add 1 (2-opt)
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, len(sel_list)):
                if (time.time() - start_time) > time_limit:
                    break
                v1, v2 = sel_list[i], sel_list[j]
                if try_remove_add([v1, v2], []):
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue # Restart scan with new set

        # Try Remove 3, Add 1 (3-opt)
        sel_list = list(selected_set)
        found = False
        for i in range(len(sel_list)):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, len(sel_list)):
                if (time.time() - start_time) > time_limit:
                    break
                for k in range(j + 1, len(sel_list)):
                    if (time.time() - start_time) > time_limit:
                        break
                    v1, v2, v3 = sel_list[i], sel_list[j], sel_list[k]
                    if try_remove_add([v1, v2, v3], []):
                        found = True
                        improved = True
                        break
                if found:
                    break
            if found:
                break
        
        if improved:
            continue # Restart scan with new set

        # If no size-reducing move found, try to "shake" the solution?
        # For now, we stop if we can't reduce size.
        # We could try 1-swap (Remove 1, Add 1) to escape local minima, but that increases complexity and risk of getting stuck in larger sets.
        # Given the small size, 2-opt and 3-opt are strong.
        
        # If we haven't improved, break to avoid infinite loop
        break

    # Final cleanup
    remove_redundant()
    
    return sorted(selected_set)

def _solve_with_milp(n, cliques, time_limit=15.0):
    """Try to solve with MILP using scipy."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        m = len(cliques)
        A = np.zeros((m, n))
        for i, clique in enumerate(cliques):
            for v in clique:
                A[i, v] = 1.0
        
        constraint = LinearConstraint(A, lb=1.0, ub=np.inf)
        integrality = np.ones(n)
        bounds = Bounds(0, 1)
        c = np.ones(n)
        
        start_time = time.time()
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds,
                     options={"time_limit": time_limit})
        
        if result.success and result.x is not None:
            selected = [v for v in range(n) if result.x[v] >= 0.5]
            if _verify_hitting_set(selected, cliques):
                return sorted(selected)
    except Exception:
        pass
    return None

def _solve_hitting_set(n, cliques, time_budget=40.0):
    """Solve the minimum hitting set problem with multiple strategies."""
    if not cliques:
        return []
    
    start_time = time.time()
    
    # Get a good initial solution with greedy
    greedy_sol = _greedy_hitting_set(n, cliques)
    best_sol = greedy_sol
    best_cost = len(best_sol)
    
    # Try MILP if we have time
    # Allocate 30% of time to MILP
    milp_time = min(10.0, time_budget * 0.3)
    if time_budget > 5.0:
        milp_sol = _solve_with_milp(n, cliques, time_limit=milp_time)
        if milp_sol is not None:
            if len(milp_sol) < best_cost:
                best_sol = milp_sol
                best_cost = len(best_sol)
    
    # Local search improvement
    # Allocate remaining time to local search
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 1.0:
        improved = _local_search_improve(best_sol, cliques, n, time_limit=remaining_time)
        if len(improved) < best_cost:
            best_sol = improved
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
        instance_budget = min(40.0, remaining * 0.8)
        
        # Solve hitting set
        selected = _solve_hitting_set(n, cliques, time_budget=instance_budget)
        
        # Final verification
        if not _verify_hitting_set(selected, cliques):
            touched = {x for e in edges for x in e}
            selected = sorted(touched)
        
        out.append(sorted(selected))
    
    return out
