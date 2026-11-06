"""
Clique partition via maximal clique enumeration + exact minimum clique cover (ILP).
For each vertex, enumerate maximal cliques containing it. Then solve a set-cover
ILP to pick the minimum number of cliques that cover every edge exactly once.
Fallback: greedy + local search if ILP is too slow.
"""
import time
import random
import sys
from itertools import combinations

_START = time.time()
DEADLINE = 140.0  # safety margin from 150s

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    """Generate the same graph as the verifier."""
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _bron_kerbosch(adj, n):
    """
    Enumerate all maximal cliques in the graph using Bron-Kerbosch with pivot.
    Returns list of frozensets of vertices.
    """
    cliques = []
    
    def _bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        if P:
            pivot = max(P, key=lambda u: sum(1 for v in P if adj[u][v]))
            candidates = [v for v in P if not adj[v][pivot]]
        else:
            pivot = max(X, key=lambda u: sum(1 for v in X if adj[u][v])) if X else None
            candidates = list(P)
        
        for v in candidates:
            _bk(R | {v}, P & {w for w in P if adj[v][w]}, X & {w for w in X if adj[v][w]})
            P = P - {v}
            X = X | {v}
    
    _bk(frozenset(), frozenset(range(n)), frozenset())
    return cliques

def _all_maximal_cliques(adj, n):
    """Get all maximal cliques of size >= 2."""
    all_cliques = _bron_kerbosch(adj, n)
    return [c for c in all_cliques if len(c) >= 2]

def _greedy_clique_cover(adj, edge_list, cliques):
    """
    Greedy: repeatedly pick the clique that covers the most uncovered edges.
    Returns list of clique vertex lists.
    """
    edge_set = set(edge_list)
    uncovered = set(edge_list)
    chosen = []
    
    # Precompute edges in each clique
    clique_edges = []
    for c in cliques:
        verts = list(c)
        edges = frozenset((min(a,b), max(a,b)) for a in range(len(verts)) for b in range(a+1, len(verts)))
        clique_edges.append(edges)
    
    remaining = list(range(len(cliques)))
    
    while uncovered and remaining:
        best_idx = None
        best_count = 0
        for idx in remaining:
            count = len(clique_edges[idx] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx is None or best_count == 0:
            break
        chosen.append(cliques[best_idx])
        uncovered -= clique_edges[best_idx]
        remaining.remove(best_idx)
    
    # Any remaining edges: add as individual edges
    for e in uncovered:
        chosen.append([e[0], e[1]])
    
    return chosen

def _local_search_improve(adj, edge_list, cliques, max_time=30.0):
    """
    Local search: try to replace a clique with a smaller set of cliques,
    or merge adjacent cliques if valid.
    """
    start = time.time()
    current = [list(c) for c in cliques]
    
    # Build edge-to-clique mapping
    def build_edge_map(cliques_list):
        edge_to_clique = {}
        for ci, c in enumerate(cliques_list):
            verts = c
            for a in range(len(verts)):
                for b in range(a+1, len(verts)):
                    e = (min(verts[a], verts[b]), max(verts[a], verts[b]))
                    if e in edge_to_clique:
                        return None  # conflict
                    edge_to_clique[e] = ci
        return edge_to_clique
    
    current_map = build_edge_map(current)
    if current_map is None:
        return current
    
    rng = random.Random(123)
    
    # Precompute all maximal cliques for potential replacements
    n = max(max(c) for c in current) + 1 if current else 0
    all_cliques = _all_maximal_cliques(adj, n)
    
    # For each edge, find all cliques that contain it
    edge_to_cliques = {e: [] for e in edge_list}
    for ci, c in enumerate(all_cliques):
        verts = list(c)
        for a in range(len(verts)):
            for b in range(a+1, len(verts)):
                e = (min(verts[a], verts[b]), max(verts[a], verts[b]))
                if e in edge_to_cliques:
                    edge_to_cliques[e].append(ci)
    
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        # Try to remove a clique and re-cover its edges with other cliques
        for ci in range(len(current) - 1, -1, -1):
            if time.time() - start > max_time:
                break
            c = current[ci]
            # Edges in this clique
            c_edges = set()
            for a in range(len(c)):
                for b in range(a+1, len(c)):
                    c_edges.add((min(c[a], c[b]), max(c[a], c[b])))
            
            # Remove this clique
            current.pop(ci)
            
            # Try to cover c_edges using remaining cliques + new cliques
            # Simple: for each edge in c_edges, if it's already covered, skip
            # Otherwise, try to find a clique from all_cliques that covers it
            uncovered = set(c_edges)
            new_cliques = []
            
            for e in list(uncovered):
                # Find a clique in all_cliques that contains e and doesn't conflict
                found = False
                for aci in edge_to_cliques.get(e, []):
                    ac = all_cliques[aci]
                    ac_verts = list(ac)
                    ac_edges = set()
                    for a in range(len(ac_verts)):
                        for b in range(a+1, len(ac_verts)):
                            ac_edges.add((min(ac_verts[a], ac_verts[b]), max(ac_verts[a], ac_verts[b])))
                    # Check no conflict with current
                    conflict = False
                    for ae in ac_edges:
                        if ae in c_edges and ae not in uncovered:
                            # This edge is in c_edges but already covered by another clique
                            # Actually, we need to check against current cliques
                            pass
                    # Check against current cliques for edge conflicts
                    for ae in ac_edges:
                        # If ae is in current map and not in our new_cliques, conflict
                        in_current = False
                        for nc in new_cliques:
                            nc_verts = nc
                            for a in range(len(nc_verts)):
                                for b in range(a+1, len(nc_verts)):
                                    if (min(nc_verts[a], nc_verts[b]), max(nc_verts[a], nc_verts[b])) == ae:
                                        in_current = True
                                        break
                            if in_current:
                                break
                        if in_current:
                            break
                        # Check against remaining current
                        for rci, rc in enumerate(current):
                            rc_verts = rc
                            for a in range(len(rc_verts)):
                                for b in range(a+1, len(rc_verts)):
                                    if (min(rc_verts[a], rc_verts[b]), max(rc_verts[a], rc_verts[b])) == ae:
                                        conflict = True
                                        break
                                if conflict:
                                    break
                            if conflict:
                                break
                        if conflict:
                            break
                    if not conflict:
                        new_cliques.append(ac)
                        uncovered -= ac_edges
                        found = True
                        break
                if not found:
                    # Add as individual edge
                    new_cliques.append([e[0], e[1]])
                    uncovered.discard(e)
            
            if len(new_cliques) < 1:  # We removed 1 clique, added 0
                improved = True
            else:
                # Revert
                current.insert(ci, c)
    
    return current

def _solve_clique_partition(adj, edge_list):
    """
    Solve the clique partition problem.
    Returns list of cliques (vertex lists).
    """
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    
    # Enumerate all maximal cliques
    all_cliques = _all_maximal_cliques(adj, n)
    
    if not all_cliques:
        return [[u, v] for u, v in edge_list]
    
    # Greedy cover
    greedy_result = _greedy_clique_cover(adj, edge_list, all_cliques)
    
    # Try ILP for better solution if time permits
    if _remaining() > 20:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
            
            n_cliques = len(all_cliques)
            n_edges = len(edge_list)
            
            # A[i,j] = 1 if clique i contains edge j
            A = np.zeros((n_edges, n_cliques))
            for i, c in enumerate(all_cliques):
                verts = list(c)
                for a in range(len(verts)):
                    for b in range(a+1, len(verts)):
                        e = (min(verts[a], verts[b]), max(verts[a], verts[b]))
                        # Find edge index
                        for j, ej in enumerate(edge_list):
                            if ej == e:
                                A[j, i] = 1
                                break
            
            # Objective: minimize sum of x_i
            c_obj = np.ones(n_cliques)
            
            # Constraints: A @ x = 1 (each edge covered exactly once)
            lb = np.ones(n_edges)
            ub = np.ones(n_edges)
            constraint = LinearConstraint(A, lb, ub)
            
            bounds = Bounds(0, 1)
            
            result = milp(c=c_obj, constraints=constraint, integrality=np.ones(n_cliques), bounds=bounds)
            
            if result.status == 0 and result.x is not None:
                ilp_cliques = []
                for i in range(n_cliques):
                    if result.x[i] > 0.5:
                        ilp_cliques.append(sorted(all_cliques[i]))
                
                # Verify
                if _verify_partition(ilp_cliques, edge_list, adj):
                    if len(ilp_cliques) < len(greedy_result):
                        return ilp_cliques
            
        except Exception:
            pass
    
    # Local search on greedy result
    if _remaining() > 10:
        ls_result = _local_search_improve(adj, edge_list, greedy_result, max_time=min(20.0, _remaining() - 5))
        if _verify_partition(ls_result, edge_list, adj):
            return ls_result
    
    return greedy_result

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    n_edges = len(edge_list)
    edge_cover = [0] * n_edges
    edge_idx = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edge_list)}
    
    for clique in cliques:
        # Check clique validity
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                if not adj[clique[a]][clique[b]]:
                    return False
        # Mark edges
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                e = (min(clique[a], clique[b]), max(clique[a], clique[b]))
                if e in edge_idx:
                    edge_cover[edge_idx[e]] += 1
    
    for c in edge_cover:
        if c != 1:
            return False
    return True

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 15:
            # Fallback: just output each edge as its own clique
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        # Solve
        cliques = _solve_clique_partition(adj, edge_list)
        
        # Verify
        if not _verify_partition(cliques, edge_list, adj):
            # Fallback: each edge its own clique
            cliques = [[u, v] for u, v in edge_list]
        
        results.append(cliques)
    
    return results
