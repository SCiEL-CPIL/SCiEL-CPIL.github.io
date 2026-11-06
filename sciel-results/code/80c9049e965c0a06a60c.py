"""
Clique partition via direct edge-assignment local search.
Representation: each edge is assigned a clique-label.
Moves: 1) move an edge to a different clique, 2) swap two edges between cliques,
       3) 3-opt rotation.
Multi-start with different random initial assignments.
"""
import time
import random
import sys
from collections import defaultdict

_START = time.time()
DEADLINE = 145.0  # safety margin from 150s

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _build_adj_set(adj, n):
    """Build set of edges for O(1) lookup."""
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edges.add((u, v))
    return edges

def _is_clique(verts, adj):
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            if not adj[verts[i]][verts[j]]:
                return False
    return True

def _random_initial_assignment(edge_list, adj, n, rng):
    """
    Create a random valid clique partition.
    Strategy: assign each edge to a random existing clique or create new one,
    then repair to ensure validity.
    """
    # Start with each edge as its own clique
    # Then try to merge edges into cliques
    # Better: use a greedy approach to build cliques
    
    # First, find all edges
    # Assign each edge to a clique label
    # Initially each edge is its own clique
    assignment = list(range(len(edge_list)))  # edge i is in clique i
    clique_edges = defaultdict(set)
    for i, e in enumerate(edge_list):
        clique_edges[i].add(i)
    
    # Now try to merge: for each pair of edges that share a vertex and form a triangle,
    # merge their cliques
    # This is complex, so instead use a simpler approach:
    # Greedy: build cliques by adding edges one by one
    
    # Actually, let's use a different approach:
    # 1. Start with no cliques
    # 2. For each edge, try to add it to an existing clique (if it maintains clique property)
    # 3. If not, create a new clique
    
    # But this might not give a valid partition because edges in a clique must all be pairwise adjacent
    # Let me think more carefully.
    
    # A valid clique partition: each clique is a set of vertices where every pair is an edge.
    # The edges in a clique are ALL pairs of vertices in that clique.
    # So if a clique has vertices {a,b,c}, it covers edges (a,b), (a,c), (b,c).
    
    # So the assignment is: each edge belongs to exactly one clique, and for each clique,
    # the set of vertices involved must form a clique in the graph.
    
    # Let me use a different representation:
    # Each clique is defined by its vertex set.
    # The edges covered by a clique are all pairs in that vertex set.
    # We need: every edge is covered by exactly one clique.
    
    # For local search, let's work with clique vertex sets.
    # Initial solution: greedy clique cover
    
    # Greedy: repeatedly find a maximal clique, assign all its uncovered edges to it
    uncovered = set(edge_list)
    cliques = []
    
    while uncovered:
        # Find a large clique among remaining edges
        # Pick a random edge, try to extend to a clique
        best_clique = None
        best_size = 0
        for e in list(uncovered)[:50]:  # sample
            u, v = e
            # Try to extend {u,v} to a larger clique
            candidates = [u, v]
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w] and adj[v][w]:
                    # Check if w can be added: all pairs in candidates+{w} must be edges
                    can_add = True
                    for x in candidates:
                        if x != w and not adj[x][w]:
                            can_add = False
                            break
                    if can_add:
                        candidates.append(w)
            
            if len(candidates) > best_size:
                best_size = len(candidates)
                best_clique = candidates
        
        if best_clique is None or best_size < 2:
            # Just take one edge
            e = next(iter(uncovered))
            best_clique = [e[0], e[1]]
        
        # Add all edges in this clique that are uncovered
        clique_set = set(best_clique)
        for a in range(len(best_clique)):
            for b in range(a+1, len(best_clique)):
                e = (min(best_clique[a], best_clique[b]), max(best_clique[a], best_clique[b]))
                if e in uncovered:
                    uncovered.remove(e)
        
        cliques.append(sorted(best_clique))
    
    return cliques

def _verify_partition(cliques, edge_list, adj):
    """Verify every edge is in exactly one clique and each clique is valid."""
    n_edges = len(edge_list)
    edge_idx = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(edge_list)}
    edge_cover = [0] * n_edges
    
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

def _clique_edges(clique):
    """Return set of edges in a clique."""
    edges = set()
    for a in range(len(clique)):
        for b in range(a+1, len(clique)):
            edges.add((min(clique[a], clique[b]), max(clique[a], clique[b])))
    return edges

def _local_search(cliques, edge_list, adj, n, max_time=20.0, rng=None):
    """
    Local search on clique partition.
    Moves:
    1. Remove a vertex from a clique (if the remaining vertices still form a valid assignment)
    2. Add a vertex to a clique (if it forms a clique and doesn't create conflicts)
    3. Move a vertex from one clique to another
    4. Split a clique into two
    5. Merge two cliques if they form a clique together
    
    Actually, let's think in terms of edge assignments.
    A move that reduces the number of cliques:
    - Merge two cliques if their union of vertices forms a clique
    - Remove a clique if its edges can be covered by other cliques (but they can't, by definition)
    
    A move that keeps the same number:
    - Restructure: take some edges from clique A, some from clique B, form new cliques
    
    Let me use a simpler approach:
    - Try to merge any pair of cliques if their vertex union is a clique
    - Try to split a large clique into two smaller cliques if that reduces total count (it won't)
    
    The key insight: we want to MINIMIZE the number of cliques.
    So the best move is to merge two cliques when possible.
    """
    if rng is None:
        rng = random.Random(42)
    
    start = time.time()
    current = [list(c) for c in cliques]
    best = [list(c) for c in current]
    best_count = len(current)
    
    # Precompute edge set for quick lookup
    edge_set = set(edge_list)
    
    # Build vertex-to-cliques mapping
    def build_vertex_map(clique_list):
        vmap = defaultdict(list)
        for ci, c in enumerate(clique_list):
            for v in c:
                vmap[v].append(ci)
        return vmap
    
    # Try merging cliques
    def try_merge(clique_list):
        """Try to merge any pair of cliques. Return new list if successful, else None."""
        n_cliques = len(clique_list)
        for i in range(n_cliques):
            for j in range(i+1, n_cliques):
                # Check if union of vertices forms a clique
                union_verts = set(clique_list[i]) | set(clique_list[j])
                verts = list(union_verts)
                valid = True
                for a in range(len(verts)):
                    for b in range(a+1, len(verts)):
                        if not adj[verts[a]][verts[b]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Check that all edges in the union are covered by these two cliques
                    # and no edge is covered by another clique
                    new_clique = sorted(verts)
                    new_edges = _clique_edges(new_clique)
                    
                    # Check: edges in new_clique should be exactly the union of edges in clique i and j
                    old_edges = _clique_edges(clique_list[i]) | _clique_edges(clique_list[j])
                    if new_edges == old_edges:
                        # Valid merge
                        new_list = [list(c) for c in clique_list]
                        new_list.pop(j)
                        new_list.pop(i)
                        new_list.append(new_clique)
                        return new_list
        return None
    
    # Try to remove a vertex from a clique and add it to another
    def try_reassign(clique_list):
        """Try to move a vertex from one clique to another, reducing total clique count."""
        # This is complex. Let's try a different approach.
        # For each clique, try removing each vertex and see if the remaining edges
        # can be absorbed into other cliques.
        n_cliques = len(clique_list)
        for ci in range(n_cliques):
            c = clique_list[ci]
            if len(c) <= 2:
                continue
            for v in c:
                # Remove v from clique ci
                remaining = [x for x in c if x != v]
                if len(remaining) < 2:
                    continue
                # Check if remaining is still a clique
                valid_rem = True
                for a in range(len(remaining)):
                    for b in range(a+1, len(remaining)):
                        if not adj[remaining[a]][remaining[b]]:
                            valid_rem = False
                            break
                    if not valid_rem:
                        break
                if not valid_rem:
                    continue
                
                # Edges that were in ci but involve v
                removed_edges = set()
                for u in remaining:
                    removed_edges.add((min(u, v), max(u, v)))
                
                # Check if these edges can be covered by other cliques
                # They must not be in any other clique (by partition property)
                # So we need to find a clique among the remaining cliques that can absorb them
                # OR create a new clique for them
                
                # Try to add v to an existing other clique
                for cj in range(n_cliques):
                    if cj == ci:
                        continue
                    other = clique_list[cj]
                    new_other = sorted(set(other) | {v})
                    # Check if new_other is a clique
                    valid_new = True
                    for a in range(len(new_other)):
                        for b in range(a+1, len(new_other)):
                            if not adj[new_other[a]][new_other[b]]:
                                valid_new = False
                                break
                    if not valid_new:
                        continue
                    
                    # Check edge coverage
                    new_other_edges = _clique_edges(new_other)
                    rem_edges = _clique_edges(remaining)
                    
                    # The edges in new_other that weren't in other must be the removed_edges
                    added_edges = new_other_edges - _clique_edges(other)
                    if added_edges == removed_edges:
                        # Valid move
                        new_list = [list(x) for x in clique_list]
                        new_list[ci] = remaining
                        new_list[cj] = new_other
                        # Check if we can remove any empty cliques
                        new_list = [x for x in new_list if len(x) >= 2]
                        return new_list
                
                # If v can't be moved, try to see if the edges involving v
                # can form their own clique with some other edges
                # This would increase clique count, so skip
        
        return None
    
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        
        # Try merges
        merged = try_merge(current)
        if merged is not None and len(merged) < len(current):
            current = merged
            improved = True
            if len(current) < best_count:
                best = [list(c) for c in current]
                best_count = len(current)
            continue
        
        # Try reassignments
        reassigned = try_reassign(current)
        if reassigned is not None and len(reassigned) < len(current):
            current = reassigned
            improved = True
            if len(current) < best_count:
                best = [list(c) for c in current]
                best_count = len(current)
            continue
        
        # Try: for each pair of cliques, see if we can restructure them
        # into fewer cliques
        n_cliques = len(current)
        for i in range(n_cliques):
            for j in range(i+1, n_cliques):
                if time.time() - start > max_time:
                    break
                # Take the union of edges from cliques i and j
                ci = current[i]
                cj = current[j]
                all_edges = _clique_edges(ci) | _clique_edges(cj)
                all_verts = set(ci) | set(cj)
                
                # Try to partition these edges into 1 or 2 cliques
                # 1 clique: check if all_verts is a clique
                if _is_clique(list(all_verts), adj):
                    # Already handled by try_merge
                    pass
                else:
                    # Try 2 cliques: see if we can split into two valid cliques
                    # with fewer total vertices (same count, but maybe enables future merges)
                    # This is complex, skip for now
                    pass
            if time.time() - start > max_time:
                break
        
        # If no improvement, try random perturbation
        if not improved:
            # Randomly restructure: pick two cliques, merge their vertices,
            # then greedily re-partition
            if len(current) >= 2:
                i, j = rng.sample(range(len(current)), 2)
                combined_verts = sorted(set(current[i]) | set(current[j]))
                combined_edges = _clique_edges(combined_verts)
                
                # Only proceed if all combined_edges are actually in the graph
                # and were covered by cliques i and j
                original_edges = _clique_edges(current[i]) | _clique_edges(current[j])
                if combined_edges == original_edges:
                    # Greedily re-partition combined_verts into cliques
                    # Try to make one big clique first
                    if _is_clique(combined_verts, adj):
                        new_list = [list(c) for c in current if c not in (current[i], current[j])]
                        new_list.append(combined_verts)
                        if len(new_list) < len(current):
                            current = new_list
                            improved = True
                            if len(current) < best_count:
                                best = [list(c) for c in current]
                                best_count = len(current)
                    else:
                        # Try to split into two cliques
                        # Greedy: pick a vertex, extend to maximal clique
                        remaining = set(combined_verts)
                        parts = []
                        while remaining:
                            v = min(remaining)
                            clique_verts = [v]
                            remaining.remove(v)
                            for w in list(remaining):
                                if all(adj[w][x] for x in clique_verts):
                                    clique_verts.append(w)
                                    remaining.remove(w)
                            if len(clique_verts) >= 2:
                                parts.append(sorted(clique_verts))
                        
                        if len(parts) < 2:
                            # Can't split into valid cliques, revert
                            pass
                        else:
                            new_list = [list(c) for c in current if c not in (current[i], current[j])]
                            new_list.extend(parts)
                            if len(new_list) < len(current):
                                current = new_list
                                improved = True
                                if len(current) < best_count:
                                    best = [list(c) for c in current]
                                    best_count = len(current)
    
    return best

def _greedy_clique_cover(adj, edge_list, n):
    """
    Greedy: repeatedly pick the largest clique among uncovered edges.
    """
    uncovered = set(edge_list)
    cliques = []
    
    while uncovered and _remaining() > 5:
        # Find the largest clique we can form from uncovered edges
        best_clique = None
        best_size = 0
        
        # Sample edges to find good starting points
        sample_edges = list(uncovered)
        rng = random.Random(hash(tuple(sorted(uncovered))) % 10000)
        rng.shuffle(sample_edges)
        sample_edges = sample_edges[:min(100, len(sample_edges))]
        
        for e in sample_edges:
            u, v = e
            # Try to extend {u, v} to a maximal clique
            candidates = {u, v}
            changed = True
            while changed:
                changed = False
                for w in range(n):
                    if w in candidates:
                        continue
                    # Check if w is adjacent to all in candidates
                    if all(adj[w][x] for x in candidates):
                        # Check if all edges from w to candidates are uncovered
                        all_uncovered = True
                        for x in candidates:
                            edge = (min(w, x), max(w, x))
                            if edge not in uncovered:
                                all_uncovered = False
                                break
                        if all_uncovered:
                            candidates.add(w)
                            changed = True
            
            if len(candidates) > best_size:
                best_size = len(candidates)
                best_clique = sorted(candidates)
        
        if best_clique is None or best_size < 2:
            # Take any edge
            e = next(iter(uncovered))
            best_clique = sorted([e[0], e[1]])
        
        # Remove all edges in this clique from uncovered
        clique_set = set(best_clique)
        for a in range(len(best_clique)):
            for b in range(a+1, len(best_clique)):
                e = (min(best_clique[a], best_clique[b]), max(best_clique[a], best_clique[b]))
                uncovered.discard(e)
        
        cliques.append(best_clique)
    
    # Any remaining edges
    for e in uncovered:
        cliques.append(sorted([e[0], e[1]]))
    
    return cliques

def _solve_clique_partition(adj, edge_list):
    """
    Solve the clique partition problem.
    Returns list of cliques (vertex lists).
    """
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    
    if not edge_list:
        return []
    
    # Greedy initial solution
    greedy_result = _greedy_clique_cover(adj, edge_list, n)
    
    # Local search with multiple restarts
    best_result = greedy_result
    best_count = len(greedy_result)
    
    start = time.time()
    max_time = min(100.0, _remaining() - 10)
    
    # Multiple restarts with different random seeds
    for restart in range(10):
        if time.time() - start > max_time or _remaining() < 15:
            break
        
        if restart == 0:
            initial = greedy_result
        else:
            # Perturb the greedy result
            initial = [list(c) for c in greedy_result]
            rng = random.Random(restart * 123 + 456)
            # Randomly merge some cliques and re-split
            if len(initial) >= 3:
                for _ in range(min(3, len(initial) - 2)):
                    i, j = rng.sample(range(len(initial)), 2)
                    combined = sorted(set(initial[i]) | set(initial[j]))
                    if _is_clique(combined, adj):
                        initial.pop(j)
                        initial.pop(i)
                        initial.append(combined)
        
        # Local search
        ls_result = _local_search(initial, edge_list, adj, n, 
                                   max_time=min(8.0, max_time - (time.time() - start)),
                                   rng=random.Random(restart * 789))
        
        if _verify_partition(ls_result, edge_list, adj):
            if len(ls_result) < best_count:
                best_count = len(ls_result)
                best_result = ls_result
    
    return best_result

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
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
