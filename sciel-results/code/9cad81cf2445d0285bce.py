"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random
import numpy as np

def _generate_graph(n, p, seed):
    """Generate graph edges using the same RNG sequence as the verifier."""
    rng = _random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot to minimize branching
        # Pivot should be in P ∪ X
        U = P | X
        if not U:
            return
        # Pick pivot with maximum degree in U
        pivot = max(U, key=lambda v: len(adj[v] & U))
        # Branch on vertices in P \ N(pivot)
        candidates = P - adj[pivot]
        for v in list(candidates):
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R_new, P_new, X_new)
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _greedy_hitting_set(cliques, n):
    """Greedy set cover: repeatedly pick vertex that covers most uncovered cliques."""
    # Map each vertex to list of clique indices it belongs to
    vertex_to_cliques = {v: [] for v in range(n)}
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].append(i)
    
    uncovered = set(range(len(cliques)))
    selected = set()
    
    while uncovered:
        # Find vertex covering most uncovered cliques
        best_v = None
        best_count = -1
        for v in range(n):
            count = sum(1 for ci in vertex_to_cliques[v] if ci in uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count == 0:
            break
        selected.add(best_v)
        for ci in vertex_to_cliques[best_v]:
            uncovered.discard(ci)
    
    return selected

def _local_search(selected, cliques, n, adj, time_limit):
    """Local search: try deletions and 2-opt swaps to reduce size."""
    start_time = time.time()
    selected = set(selected)
    
    # Precompute for each vertex, which cliques it covers
    def vertex_clique_coverage(v):
        return [i for i, c in enumerate(cliques) if v in c]
    
    # Try deletions
    changed = True
    while changed:
        changed = False
        for v in list(selected):
            # Check if removing v still covers all cliques
            # v is needed if there exists a clique where v is the only selected vertex
            needed = False
            for c in cliques:
                if v in c:
                    # Check if any other selected vertex is in this clique
                    if not any(other in c for other in selected - {v}):
                        needed = True
                        break
            if not needed:
                selected.discard(v)
                changed = True
                if time.time() - start_time > time_limit:
                    return selected
    
    # Try 2-opt swaps: remove one vertex, add another, if size doesn't increase and coverage maintained
    # Actually, swaps only help if we can remove 1 and add 1 but that keeps size same.
    # Better: try to replace a vertex with another that covers a superset of its unique cliques.
    # Or: try adding a vertex that allows removing two vertices.
    
    # Try: for each pair of vertices in selected, see if one can replace both
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit:
            return selected
        selected_list = list(selected)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                v1, v2 = selected_list[i], selected_list[j]
                # Try to find a vertex w not in selected that covers all cliques that v1 or v2 uniquely cover
                # Cliques that need v1 or v2 (i.e., cliques where the only selected vertices are subset of {v1, v2})
                critical_cliques = set()
                for ci, c in enumerate(cliques):
                    sel_in_c = [x for x in c if x in selected]
                    if v1 in c and v2 in c:
                        # This clique is covered by both
                        pass
                    elif v1 in c:
                        # Only v1 from {v1,v2} is in this clique
                        if len(sel_in_c) == 1:
                            critical_cliques.add(ci)
                    elif v2 in c:
                        if len(sel_in_c) == 1:
                            critical_cliques.add(ci)
                
                if not critical_cliques:
                    continue
                
                # Find a vertex w not in selected that covers all critical_cliques
                for w in range(n):
                    if w in selected:
                        continue
                    covers_all = all(w in cliques[ci] for ci in critical_cliques)
                    if covers_all:
                        # Check that after removing v1, v2 and adding w, all cliques are still covered
                        new_selected = (selected - {v1, v2}) | {w}
                        # Verify coverage
                        all_covered = True
                        for c in cliques:
                            if not any(x in c for x in new_selected):
                                all_covered = False
                                break
                        if all_covered:
                            selected = new_selected
                            changed = True
                            break
                if changed:
                    break
            if changed:
                break
    
    return selected

def _milp_solve(cliques, n, time_limit):
    """Try to solve the hitting set problem using MILP via scipy."""
    m = len(cliques)
    if m == 0:
        return set()
    
    # Build constraint matrix
    A = np.zeros((m, n), dtype=np.int8)
    for i, clique in enumerate(cliques):
        for v in clique:
            A[i, v] = 1
    
    c = np.ones(n, dtype=np.float64)
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraints = LinearConstraint(A, 1, np.inf)
        bounds_obj = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=np.ones(n, dtype=np.int8), 
                   bounds=bounds_obj, 
                   options={"time_limit": time_limit, "presolve": True})
        
        if res.x is not None:
            sol = set(i for i in range(n) if res.x[i] > 0.5)
            # Verify validity
            valid = True
            for c_clique in cliques:
                if not any(v in c_clique for v in sol):
                    valid = False
                    break
            if valid:
                return sol
    except Exception:
        pass
    
    return None

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline - 30.0:
            # Not enough time left; return trivial solution for remaining
            adj = _generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue
        
        adj = _generate_graph(n, p, seed)
        
        # Enumerate maximal cliques
        cliques = _bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # First try MILP with a reasonable time budget
        time_left = deadline - time.time() - 10.0
        milp_budget = min(20.0, time_left * 0.5)  # Give MILP up to half the remaining time, capped at 20s
        
        selected = _milp_solve(cliques, n, milp_budget)
        
        if selected is None:
            # Fall back to greedy
            selected = _greedy_hitting_set(cliques, n)
        
        # Always run local search to polish
        time_left = deadline - time.time() - 10.0
        if time_left > 1.0:
            selected = _local_search(selected, cliques, n, adj, time_left)
        
        results.append(sorted(selected))
    
    return results
# EVOLVE-BLOCK-END
