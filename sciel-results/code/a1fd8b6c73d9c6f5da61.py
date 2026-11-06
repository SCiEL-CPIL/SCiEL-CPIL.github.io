"""
Solves the Minimum Vertex Cover problem for three fixed random graphs.
The problem is equivalent to finding a set of vertices that intersects every maximal clique.
Since every edge is in a maximal clique, and a vertex cover covers all edges, 
a Minimum Vertex Cover is the optimal solution.
We solve this by finding a Maximum Independent Set (MIS) in the graph, 
as Min Vertex Cover = V \ Max Independent Set.
"""

import time
import random
import sys

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

def _max_independent_set(adj, deadline):
    """
    Finds a Maximum Independent Set in the graph defined by adj.
    Uses a recursive backtracking approach with pruning.
    """
    n = len(adj)
    # Convert adj to a list of sets for faster intersection if needed, 
    # but we will use bitmasks or sets. For n<=40, sets are fine.
    
    # We want to maximize the size of the independent set.
    best_size = 0
    best_set = set()
    
    # Pre-compute degrees to help with ordering? 
    # A simple heuristic: sort vertices by degree descending initially?
    # Or just use the standard Tomita pivot strategy adapted for MIS.
    # Actually, for MIS, a common effective strategy is to branch on the vertex 
    # with the highest degree in the remaining candidate set P.
    
    # To ensure we hit the deadline safely, we track time.
    
    def expand(current_set, candidates):
        nonlocal best_size, best_set
        
        # Pruning 1: If we can't beat the best size, return
        if len(current_set) + len(candidates) <= best_size:
            return
            
        # Check time
        if time.time() > deadline:
            return

        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return

        # Choose a vertex to branch on.
        # Heuristic: Pick the vertex in candidates with the highest degree 
        # within the candidates set. This tends to reduce the candidate set 
        # more aggressively when included.
        
        # Calculate degrees within candidates
        # This is O(|P|^2) per node, which is fine for small n.
        # Optimization: We can just pick an arbitrary vertex if we want, 
        # but degree heuristic helps a lot.
        
        # Let's find the vertex with max degree in candidates
        max_deg = -1
        pivot = None
        for v in candidates:
            deg = len(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            # Should not happen if candidates is not empty
            return

        # Branch 1: Include pivot
        # New candidates must be neighbors of pivot in the original graph?
        # NO. Independent set: if we include v, we cannot include its neighbors.
        # So new candidates = candidates \ N(v)
        # Wait, the definition of Independent Set: no two vertices are adjacent.
        # If we add v to the set, we must remove all neighbors of v from candidates.
        
        new_candidates_include = candidates - adj[pivot]
        # Note: pivot itself is removed from candidates implicitly as it's added to current_set
        expand(current_set | {pivot}, new_candidates_include)
        
        # Branch 2: Exclude pivot
        # We just remove pivot from candidates
        new_candidates_exclude = candidates - {pivot}
        expand(current_set, new_candidates_exclude)

    # Initial call
    # We can try to order vertices initially to help, but the recursive 
    # pivot selection handles the dynamic ordering.
    initial_candidates = set(range(n))
    
    # Optional: Initial lower bound to help pruning
    # A greedy independent set can provide a good initial bound.
    def greedy_mis(adj, n):
        remaining = set(range(n))
        mis = set()
        while remaining:
            # Pick vertex with min degree in remaining? Or just arbitrary?
            # Min degree is often better for packing.
            v = min(remaining, key=lambda x: len(adj[x] & remaining))
            mis.add(v)
            remaining -= adj[v]
            remaining.discard(v)
        return mis

    init_mis = greedy_mis(adj, n)
    best_size = len(init_mis)
    best_set = init_mis
    
    expand(set(), initial_candidates)
    
    return best_set

def _greedy_vertex_cover(adj, deadline):
    """Fallback greedy vertex cover if exact solver fails or times out."""
    n = len(adj)
    cover = set()
    remaining_edges = [(u, v) for u in range(n) for v in range(u+1, n) if v in adj[u]]
    
    while remaining_edges and time.time() < deadline:
        # Find vertex covering most remaining edges
        best_v = None
        best_count = 0
        for v in range(n):
            if v in cover:
                continue
            count = sum(1 for (u, w) in remaining_edges if u == v or w == v)
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v is None:
            break
            
        cover.add(best_v)
        # Remove edges covered by best_v
        remaining_edges = [(u, w) for (u, w) in remaining_edges if u != best_v and w != best_v]
        
    return sorted(cover)

def construct():
    start_time = time.time()
    # Leave 5 seconds for safety
    deadline = start_time + 145.0
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline - 10:
            # Not enough time, append empty or greedy if possible quickly
            # For safety, just append empty to avoid timeout, 
            # though the exact solver should be very fast for these sizes.
            results.append([])
            continue
            
        adj = _build_graph(n, p, seed)
        
        # Try exact solver
        try:
            mis = _max_independent_set(adj, deadline - 5)
            # Vertex Cover is complement of MIS
            vc = set(range(n)) - mis
            sol = sorted(vc)
            
            # Verify validity
            is_valid = True
            for u in range(n):
                for v in adj[u]:
                    if u > v: # Check each edge once
                        if u not in vc and v not in vc:
                            is_valid = False
                            break
                if not is_valid:
                    break
            
            if not is_valid:
                # Fallback if something went wrong (should not happen)
                sol = _greedy_vertex_cover(adj, deadline - 2)
                
            results.append(sol)
            
        except Exception:
            # Fallback to greedy
            sol = _greedy_vertex_cover(adj, deadline - 2)
            results.append(sol)
            
    return results

# EVOLVE-BLOCK-END
