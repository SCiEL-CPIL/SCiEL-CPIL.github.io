"""
Clique partition via greedy clique extraction + local search.
Strategy:
1. Repeatedly find a large clique in the remaining graph, cover all its edges,
   remove those edges, repeat until no edges remain.
2. Local search: try to move an edge from one clique to another if valid,
   or split a clique into two smaller cliques to free up edges for merging.
"""
import time
import random
from itertools import combinations
from collections import defaultdict

_START = time.time()
DEADLINE = 145.0  # safety margin from 150s

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

def _is_clique(verts, adj):
    """Check if a set of vertices forms a clique."""
    verts_list = list(verts)
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if not adj[verts_list[i]][verts_list[j]]:
                return False
    return True

def _find_large_clique(adj, remaining_edges, n):
    """
    Find a relatively large clique in the subgraph induced by remaining_edges.
    Use a greedy approach: start with a random edge, grow the clique.
    """
    if not remaining_edges:
        return None
    
    # Build adjacency restricted to remaining edges
    # For each vertex, which other vertices are connected via remaining edges
    vert_adj = defaultdict(set)
    for u, v in remaining_edges:
        vert_adj[u].add(v)
        vert_adj[v].add(u)
    
    # Greedy: try multiple random starting points
    best_clique = None
    rng = random.Random(123)
    
    # Get all vertices that have at least one remaining edge
    active_verts = set()
    for u, v in remaining_edges:
        active_verts.add(u)
        active_verts.add(v)
    
    if not active_verts:
        return None
    
    # Try a few random starts
    for _ in range(50):
        if _remaining() < 10:
            break
        start = rng.choice(list(active_verts))
        # Grow clique greedily
        clique = {start}
        candidates = set(vert_adj.get(start, set()))
        
        while candidates:
            # Pick candidate that maximizes common neighbors in candidates
            best_cand = None
            best_score = -1
            for cand in list(candidates):
                # Count how many other candidates are adjacent to cand
                score = len(candidates & vert_adj.get(cand, set()))
                if score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand is None:
                break
            clique.add(best_cand)
            # Update candidates: must be adjacent to all in clique
            candidates = set()
            for v in list(candidates):
                if all(adj[v][w] for w in clique if w != v):
                    candidates.add(v)
        
        if best_clique is None or len(clique) > len(best_clique):
            best_clique = clique
    
    return best_clique

def _greedy_clique_partition(adj, edge_list):
    """
    Greedy: repeatedly find a large clique, cover all its edges, remove them.
    Returns list of cliques (each is a set of vertices).
    """
    remaining = set(edge_list)
    cliques = []
    
    while remaining and _remaining() > 10:
        # Find a large clique in the remaining edges
        clique = _find_large_clique(adj, remaining, len(edge_list[0][0]) if edge_list else 0)
        if clique is None or len(clique) < 2:
            # Fallback: take one edge as a clique
            edge = next(iter(remaining))
            clique = {edge[0], edge[1]}
        
        # Cover all edges within this clique
        clique_edges = set()
        verts_list = list(clique)
        for i in range(len(verts_list)):
            for j in range(i + 1, len(verts_list)):
                e = (min(verts_list[i], verts_list[j]), max(verts_list[i], verts_list[j]))
                if e in remaining:
                    clique_edges.add(e)
        
        if not clique_edges:
            # Shouldn't happen, but safety
            edge = next(iter(remaining))
            clique = {edge[0], edge[1]}
            clique_edges = {edge}
        
        cliques.append(clique)
        remaining -= clique_edges
    
    return cliques

def _clique_edges(clique, adj):
    """Get all edges within a clique."""
    verts = list(clique)
    edges = set()
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if adj[verts[i]][verts[j]]:
                edges.add((min(verts[i], verts[j]), max(verts[i], verts[j])))
    return edges

def _local_search_improve(cliques, adj, edge_list, max_time=60.0):
    """
    Local search to reduce number of cliques.
    Moves:
    1. Move an edge from clique A to clique B if valid.
    2. Split a clique into two if it helps.
    3. Merge two cliques if their union is a clique.
    """
    n = len(edge_list[0][0]) if edge_list else 0
    start_time = time.time()
    max_time = min(max_time, _remaining() - 5)
    if max_time <= 0:
        return cliques
    
    cliques = [set(c) for c in cliques]
    rng = random.Random(42)
    
    # Precompute edge to clique mapping
    def build_edge_map(cliques, adj):
        edge_map = {}
        for ci, clique in enumerate(cliques):
            for e in _clique_edges(clique, adj):
                edge_map[e] = ci
        return edge_map
    
    edge_map = build_edge_map(cliques, adj)
    
    def try_move_edge(edge, from_clique, to_clique, cliques, adj):
        """Try to move edge from one clique to another."""
        u, v = edge
        
        # Check if edge can be added to to_clique
        new_clique = cliques[to_clique] | {u, v}
        if not _is_clique(new_clique, adj):
            return False
        
        # Check if removing edge from from_clique is valid
        # (removing an edge can't break clique property, but we need to check
        #  that the remaining vertices still form a clique)
        if from_clique != to_clique:
            old_clique = cliques[from_clique]
            # The old clique minus this edge's vertices (if they're only connected via this edge)
            # Actually, we're moving the edge, not the vertices. The vertices stay in both cliques
            # if they appear in other edges. This is tricky.
            # For simplicity, let's only allow moving if the vertices are "free"
            # i.e., they don't appear in any other edge of the from_clique
            other_edges = _clique_edges(old_clique, adj) - {edge}
            # Check if u and v still need to be in from_clique
            u_needed = any(u in e for e in other_edges)
            v_needed = any(v in e for e in other_edges)
            if not u_needed and not v_needed:
                # Can remove u and v from from_clique
                cliques[from_clique] = old_clique - {u, v}
                if len(cliques[from_clique]) < 2:
                    return False  # Can't have a clique with < 2 vertices
            else:
                # Vertices must stay, so we can't really "move" the edge in a meaningful way
                # unless we split
                return False
        
        cliques[to_clique] = new_clique
        return True
    
    def try_merge(ci, cj, cliques, adj):
        """Try to merge two cliques."""
        union = cliques[ci] | cliques[cj]
        if _is_clique(union, adj):
            cliques[ci] = union
            cliques.pop(cj)
            return True
        return False
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        
        # Try to merge pairs of cliques
        n_cliques = len(cliques)
        for i in range(n_cliques):
            for j in range(i + 1, n_cliques):
                if try_merge(i, j, cliques, adj):
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try to move edges between cliques
        # Pick a random edge and try to move it
        if edge_map:
            edge = rng.choice(list(edge_map.keys()))
            from_ci = edge_map[edge]
            for to_ci in range(len(cliques)):
                if to_ci == from_ci:
                    continue
                if try_move_edge(edge, from_ci, to_ci, cliques, adj):
                    # Remove edge from old clique if possible
                    # Rebuild edge map
                    edge_map = build_edge_map(cliques, adj)
                    improved = True
                    break
        
        if time.time() - start_time < max_time:
            # Small probability of random restart
            if rng.random() < 0.01:
                break
    
    return cliques

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    edge_cover = [0] * len(edge_list)
    edge_idx = {(min(u, v), max(u, v)): i for i, (u, v) in enumerate(edge_list)}
    
    for clique in cliques:
        # Check clique validity
        verts = list(clique)
        for a in range(len(verts)):
            for b in range(a + 1, len(verts)):
                if not adj[verts[a]][verts[b]]:
                    return False
        # Mark edges
        for a in range(len(verts)):
            for b in range(a + 1, len(verts)):
                e = (min(verts[a], verts[b]), max(verts[a], verts[b]))
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
        
        # Greedy clique extraction
        cliques = _greedy_clique_partition(adj, edge_list)
        
        # Local search improvement
        cliques = _local_search_improve(cliques, adj, edge_list, max_time=min(60.0, _remaining() - 10))
        
        # Verify
        if not _verify_partition(cliques, edge_list, adj):
            # Fallback: each edge its own clique
            cliques = [[u, v] for u, v in edge_list]
        
        results.append([sorted(c) for c in cliques])
    
    return results
