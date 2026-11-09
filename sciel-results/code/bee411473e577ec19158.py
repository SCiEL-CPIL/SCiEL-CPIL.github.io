import time
from random import Random
from collections import defaultdict

def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _is_clique(edge_set, adj=None):
    """Check if a set of edges forms a valid clique (size >= 2 vertices)."""
    if not edge_set:
        return False
    
    vertices = set()
    for u, v in edge_set:
        vertices.add(u)
        vertices.add(v)
        
    if len(vertices) < 2:
        return False
        
    # If the number of edges equals the max possible edges for |V| vertices,
    # and all edges are distinct (which they are in a set), it's a complete graph.
    expected_edges = len(vertices) * (len(vertices) - 1) // 2
    return len(edge_set) == expected_edges

def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Initial Solution: Greedy Triangle Extraction
    remaining_edges = set(edges)
    cliques = [] # List of sets of edges
    
    # Helper to find a maximal clique starting from an edge
    def expand_clique(start_edge):
        u, v = start_edge
        clique_edges = {start_edge}
        clique_vertices = {u, v}
        
        # Candidates are common neighbors
        candidates = adj[u] & adj[v]
        candidates.difference_update(clique_vertices)
        
        # Sort by degree in remaining graph to prioritize "dense" connections
        # This helps form larger cliques
        def deg(w):
            return len(adj[w] & clique_vertices) # Heuristic: how many clique members is it connected to?
            # Actually, we just need it to connect to ALL.
            # Let's just iterate.
        
        # Greedy add
        # To be deterministic and fast, we can just pick any valid one.
        # Randomizing might help escape local minima, but for init, greedy is fine.
        added = True
        while added:
            added = False
            for w in list(candidates):
                # Check if w connects to all in clique_vertices
                # Since w is in adj[u] & adj[v], it connects to u and v.
                # We need to check against all other vertices in clique_vertices.
                valid = True
                for x in clique_vertices:
                    if w not in adj[x]:
                        valid = False
                        break
                if valid:
                    # Add w
                    clique_vertices.add(w)
                    candidates.remove(w)
                    # Add all edges from w to clique_vertices
                    for x in clique_vertices:
                        if x == w: continue
                        e = (min(w, x), max(w, x))
                        if e in remaining_edges:
                            clique_edges.add(e)
                    added = True
                    break # Restart candidate loop as set changed? 
                    # Actually, if we add w, other candidates might become invalid if they don't connect to w.
                    # But we only add w if it connects to ALL.
                    # So other candidates still connect to old clique, but do they connect to w?
                    # If they don't, they can't be added *after* w in the same clique unless they connect to w.
                    # So we should filter candidates.
                    candidates = {c for c in candidates if w in adj[c]}
                    break
        
        return clique_edges

    while remaining_edges:
        if time.time() > deadline:
            break
            
        # Pick an edge to start a clique
        # Heuristic: Pick edge with most common neighbors in remaining graph
        # For speed, just pick first or random
        # Let's pick random for variety in init
        e = next(iter(remaining_edges))
        
        clique_edges = expand_clique(e)
        
        # Remove from remaining
        remaining_edges.difference_update(clique_edges)
        
        if clique_edges:
            cliques.append(clique_edges)

    # If time ran out during init, just add remaining as single edges
    for e in remaining_edges:
        cliques.append({e})
    
    # 2. Local Search: Merge and Move
    # Map edge to clique index
    edge_to_clique = {}
    for i, c in enumerate(cliques):
        for e in c:
            edge_to_clique[e] = i
    
    rng = Random(42)
    
    def try_merge(i, j):
        """Try to merge clique i and j. Returns True if merged."""
        if i == j:
            return False
        c_i = cliques[i]
        c_j = cliques[j]
        
        # Check if union is a clique
        if _is_clique(c_i | c_j):
            # Merge j into i
            c_i.update(c_j)
            for e in c_j:
                edge_to_clique[e] = i
            # Remove clique j
            cliques[j] = set() # Mark as empty
            # Update indices? 
            # To keep it simple, we don't shift indices, we just skip empty ones in future.
            # Or we can compact later.
            return True
        return False

    def try_move_edge(e, target_clique_idx):
        """Try to move edge e to target_clique_idx. Returns True if moved."""
        src_idx = edge_to_clique[e]
        if src_idx == target_clique_idx:
            return False
        if not cliques[target_clique_idx]: # Target is empty
            return False
            
        c_src = cliques[src_idx]
        c_tgt = cliques[target_clique_idx]
        
        # Check if c_tgt U {e} is a clique
        if _is_clique(c_tgt | {e}):
            # Check if c_src - {e} is still a valid clique (or empty)
            # c_src - {e} is always a subgraph of a clique, so it's a clique.
            # But it must have >= 2 vertices?
            # If c_src had 1 edge (size 2 vertices), removing it leaves 0 vertices.
            # An empty clique is not a valid "listed set".
            # So if c_src becomes empty, we must handle it.
            
            # Perform move
            c_src.remove(e)
            c_tgt.add(e)
            edge_to_clique[e] = target_clique_idx
            
            # If c_src is now empty, mark it as such
            # We don't delete from list to keep indices stable, just leave as empty set
            return True
        return False

    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        
        # Pass 1: Try all merges
        n_cliques = len(cliques)
        for i in range(n_cliques):
            if not cliques[i]:
                continue
            for j in range(i + 1, n_cliques):
                if not cliques[j]:
                    continue
                if try_merge(i, j):
                    improved = True
                    # After merge, break inner loop to re-evaluate? 
                    # Or just continue. Merging j into i means j is empty.
                    # We can break to restart the scan from 0 to catch new merges involving i.
                    break
            if improved:
                break
        
        # Pass 2: Perturbation (Move edges)
        if not improved:
            # Pick random edge
            if not edges:
                break
            e = rng.choice(edges)
            src_idx = edge_to_clique[e]
            if not cliques[src_idx]:
                continue
                
            # Pick random target clique
            # We should pick a non-empty clique
            non_empty_cliques = [i for i in range(n_cliques) if cliques[i]]
            if len(non_empty_cliques) <= 1:
                break
            target_idx = rng.choice(non_empty_cliques)
            
            if try_move_edge(e, target_idx):
                improved = True # This move might enable a merge, so loop again

    # Compact the cliques list
    final_cliques = [c for c in cliques if c]
    
    # Convert to list of lists of vertices? 
    # The task says: "two lists of cliques (vertex lists)"
    # "Every listed set must be a clique with at least two vertices"
    # So we return the vertex sets, not edge sets.
    
    result = []
    for c in final_cliques:
        vertices = set()
        for u, v in c:
            vertices.add(u)
            vertices.add(v)
        result.append(list(vertices))
        
    return result

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = _solve_instance(n, p, seed, deadline)
        result.append(cliques)
        if time.time() > deadline - 5:
            # Not enough time for second instance, return edges for second
            edges = _random_graph(n, p, seed)
            # Convert edges to vertex lists
            result.append([list(e) for e in edges])
            break
    return result
