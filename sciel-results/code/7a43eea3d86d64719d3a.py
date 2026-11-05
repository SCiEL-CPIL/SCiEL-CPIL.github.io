"""Decompose three fixed instance graphs into few cycles + single edges."""
import time
import sys
from random import Random
from collections import defaultdict, deque

_DEADLINE = 150.0
_START = None

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed() - 3.0

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycles_in_subgraph(adj, edge_set):
    """Find all simple cycles in the subgraph defined by edge_set/adj."""
    cycles = []
    visited_edges = set()
    
    for e in list(edge_set):
        if e in visited_edges:
            continue
        # Try to find a cycle containing this edge via DFS
        u, v = e
        if u not in adj or v not in adj or len(adj[u]) < 2 or len(adj[v]) < 2:
            visited_edges.add(e)
            continue
        
        # DFS to find a cycle through edge (u,v)
        # We need a path from v back to u that doesn't use edge (u,v)
        path = [u]
        path_set = {u}
        used = set()
        found = False
        
        def dfs(cur, target, depth):
            nonlocal found
            if depth > 50:  # safety
                return False
            if cur == target and len(path) >= 3:
                found = True
                return True
            for nxt in sorted(adj[cur]):
                ne = tuple(sorted((cur, nxt)))
                if ne in edge_set and ne not in used and ne != tuple(sorted((u, v))):
                    if nxt not in path_set:
                        used.add(ne)
                        path_set.add(nxt)
                        path.append(nxt)
                        if dfs(nxt, target, depth + 1):
                            return True
                        path.pop()
                        path_set.remove(nxt)
                        used.discard(ne)
            return False
        
        if dfs(v, u, 0):
            cycles.append(list(path))
            for i in range(len(path)):
                a, b = path[i], path[(i + 1) % len(path)]
                visited_edges.add(tuple(sorted((a, b))))
        else:
            visited_edges.add(e)
    
    return cycles


def _max_cycle_decomposition(edges):
    """
    Structural approach: iteratively find a maximum 2-matching (2-regular subgraph),
    extract all cycles from it, remove those edges, repeat.
    This maximizes edges covered by cycles at each step.
    """
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _adjacency(edges)
    pieces = []
    rng = Random(123)
    
    while edge_set and _time_left() > 1.0:
        # Find all edges that are in at least one cycle (have both endpoints with degree >= 2)
        cycleable_edges = set()
        for e in edge_set:
            u, v = e
            if len(adj[u]) >= 2 and len(adj[v]) >= 2:
                cycleable_edges.add(e)
        
        if not cycleable_edges:
            break
        
        # Find cycles in the cycleable subgraph
        # Build subgraph adjacency
        sub_adj = defaultdict(set)
        for e in cycleable_edges:
            u, v = e
            sub_adj[u].add(v)
            sub_adj[v].add(u)
        
        # Find all cycles using DFS on the subgraph
        cycles = []
        used_in_cycles = set()
        
        for e in sorted(cycleable_edges):
            if e in used_in_cycles:
                continue
            u, v = e
            if u not in sub_adj or v not in sub_adj:
                continue
            
            # Find a cycle through (u,v) using DFS
            path = [u]
            path_set = {u}
            used = set()
            found = False
            
            def dfs_cycle(cur, target, depth):
                nonlocal found
                if depth > len(edge_set) + 5:
                    return False
                if cur == target and len(path) >= 3:
                    found = True
                    return True
                nbrs = list(sub_adj[cur])
                rng.shuffle(nbrs)
                for nxt in nbrs:
                    ne = tuple(sorted((cur, nxt)))
                    if ne in cycleable_edges and ne not in used and ne != tuple(sorted((u, v))):
                        if nxt not in path_set:
                            used.add(ne)
                            path_set.add(nxt)
                            path.append(nxt)
                            if dfs_cycle(nxt, target, depth + 1):
                                return True
                            path.pop()
                            path_set.remove(nxt)
                            used.discard(ne)
                return False
            
            if dfs_cycle(v, u, 0):
                cycles.append(list(path))
                for i in range(len(path)):
                    a, b = path[i], path[(i + 1) % len(path)]
                    used_in_cycles.add(tuple(sorted((a, b))))
        
        if not cycles:
            # No more cycles can be found; break
            break
        
        # Remove all cycle edges from the main graph
        for cycle in cycles:
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((a, b)))
                if e in edge_set:
                    edge_set.remove(e)
                    adj[a].discard(b)
                    adj[b].discard(a)
            pieces.append(cycle)
    
    # Remaining edges as single-edge pieces
    for e in sorted(edge_set):
        pieces.append(list(e))
    
    return pieces


def _merge_singles_into_cycles(pieces, edges, time_budget):
    """
    Try to merge single-edge pieces into cycles by finding cycles in the
    residual single-edge graph.
    """
    if _time_left() < 1.0 or time_budget < 0.5:
        return pieces
    
    single_edges = [p for p in pieces if len(p) == 2]
    cycle_pieces = [p for p in pieces if len(p) >= 3]
    
    if not single_edges:
        return pieces
    
    edge_set = set(tuple(sorted(e)) for e in single_edges)
    adj = _adjacency(single_edges)
    rng = Random(456)
    deadline = time.time() + time_budget
    
    while edge_set and time.time() < deadline:
        # Find cycles in the residual single-edge graph
        cycles = []
        used_in_cycles = set()
        
        for e in sorted(edge_set):
            if e in used_in_cycles:
                continue
            u, v = e
            if len(adj[u]) < 2 or len(adj[v]) < 2:
                continue
            
            path = [u]
            path_set = {u}
            used = set()
            found = False
            
            def dfs_singles(cur, target, depth):
                nonlocal found
                if depth > 30:
                    return False
                if cur == target and len(path) >= 3:
                    found = True
                    return True
                nbrs = list(adj[cur])
                rng.shuffle(nbrs)
                for nxt in nbrs:
                    ne = tuple(sorted((cur, nxt)))
                    if ne in edge_set and ne not in used and ne != tuple(sorted((u, v))):
                        if nxt not in path_set:
                            used.add(ne)
                            path_set.add(nxt)
                            path.append(nxt)
                            if dfs_singles(nxt, target, depth + 1):
                                return True
                            path.pop()
                            path_set.remove(nxt)
                            used.discard(ne)
                return False
            
            if dfs_singles(v, u, 0):
                cycles.append(list(path))
                for i in range(len(path)):
                    a, b = path[i], path[(i + 1) % len(path)]
                    used_in_cycles.add(tuple(sorted((a, b))))
        
        if not cycles:
            break
        
        for cycle in cycles:
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((a, b)))
                if e in edge_set:
                    edge_set.remove(e)
                    adj[a].discard(b)
                    adj[b].discard(a)
            cycle_pieces.append(cycle)
    
    remaining_singles = [list(e) for e in sorted(edge_set)]
    return cycle_pieces + remaining_singles


def _local_search_pieces(pieces, edges, time_budget):
    """
    Local search: try to replace sets of single edges with cycles by
    finding augmenting cycle structures. Also try to split long cycles
    if that helps (unlikely but check).
    """
    if _time_left() < 1.0 or time_budget < 0.5:
        return pieces
    
    # Try merging pairs of adjacent single edges into 3-cycles
    single_edges = [p for p in pieces if len(p) == 2]
    cycle_pieces = [p for p in pieces if len(p) >= 3]
    
    if len(single_edges) < 3:
        return pieces
    
    edge_set = set(tuple(sorted(e)) for e in single_edges)
    adj = _adjacency(single_edges)
    rng = Random(789)
    deadline = time.time() + time_budget
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to find a 3-cycle (triangle) in the single-edge graph
        for u in sorted(adj):
            if time.time() > deadline:
                break
            for v in sorted(adj[u]):
                if time.time() > deadline:
                    break
                if v in adj[u] and u in adj[v]:
                    # Check if (u,v), (v,w), (w,u) all exist for some w
                    common = adj[u] & adj[v]
                    for w in common:
                        if w != u and w != v:
                            e1 = tuple(sorted((u, v)))
                            e2 = tuple(sorted((v, w)))
                            e3 = tuple(sorted((w, u)))
                            if e1 in edge_set and e2 in edge_set and e3 in edge_set:
                                # Found a triangle!
                                edge_set.remove(e1)
                                edge_set.remove(e2)
                                edge_set.remove(e3)
                                adj[u].discard(v)
                                adj[v].discard(u)
                                adj[v].discard(w)
                                adj[w].discard(v)
                                adj[w].discard(u)
                                adj[u].discard(w)
                                cycle_pieces.append([u, v, w])
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
    
    remaining_singles = [list(e) for e in sorted(edge_set)]
    return cycle_pieces + remaining_singles


def construct():
    global _START
    _START = time.time()
    sys.setrecursionlimit(10000)
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    all_pieces = []
    
    for spec in specs:
        if _time_left() < 3.0:
            edges = _instance(*spec)
            all_pieces.append([list(e) for e in edges])
            continue
        
        edges = _instance(*spec)
        
        # Main decomposition: extract all cycles at once
        pieces = _max_cycle_decomposition(edges)
        
        # Polish: merge remaining singles into cycles
        if _time_left() > 2.0:
            pieces = _merge_singles_into_cycles(pieces, edges, min(5.0, _time_left() * 0.3))
        
        # Local search: find triangles in remaining singles
        if _time_left() > 2.0:
            pieces = _local_search_pieces(pieces, edges, min(3.0, _time_left() * 0.2))
        
        all_pieces.append(pieces)
    
    return all_pieces
