"""Clique partition via greedy density-based set-cover."""
# EVOLVE-BLOCK-START
import time
import itertools
import heapq
from collections import defaultdict

_START = time.time()
_LIMIT = 145.0  # safety margin under 150s wall-clock


def _deadline_ok():
    return (time.time() - _START) < _LIMIT


def _elapsed():
    return time.time() - _START


def _random_edges(n, p, seed):
    """Reproduce verifier's random_graph: iterate u<v in order, rng.random()<p."""
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _solve_instance(n, p, seed):
    """
    Partition edges into minimum number of cliques.
    Strategy: Greedy set-cover.
    1. Compute 'clique density' for each edge (number of common neighbors).
    2. Sort edges by density (descending).
    3. For each edge (if still uncovered), expand it into the largest possible clique 
       using a greedy heuristic that prioritizes vertices with high connectivity to the current clique.
    4. Mark all edges in this clique as covered.
    5. Repeat until all edges are covered.
    """
    edges = _random_edges(n, p, seed)
    all_edges = set(_edge_key(u, v) for (u, v) in edges)
    
    if not all_edges:
        return []

    # Build adjacency list and sets for fast intersection
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute common neighbor counts for all edges to determine priority
    # For each edge (u, v), density = |N(u) ∩ N(v)|
    # We want to process high-density edges first to form large cliques.
    
    # Instead of precomputing all, we can just sort the initial edges by a heuristic.
    # However, as we remove edges, the "density" of remaining edges in the residual graph changes.
    # But the task is to partition the ORIGINAL graph's edges. 
    # The constraint is that each listed set must be a clique in the ORIGINAL graph?
    # "Every listed set must be a clique ... and every edge must lie in exactly one listed clique."
    # Usually "clique" implies in the graph being partitioned. The graph is fixed.
    # So a set S is a clique if every pair in S is an edge in the original graph.
    
    # Let's use a greedy approach on the set of all edges.
    # Priority: Edges that are part of many triangles (high common neighbors) are likely part of large cliques.
    
    # Compute common neighbors for all edges once.
    # This is O(E * deg) or similar. For n=64, E ~ 1000. This is fast.
    edge_density = {}
    for u, v in edges:
        key = _edge_key(u, v)
        # Intersection of adj[u] and adj[v]
        # Since n is small, we can use bitmasks for speed if needed, but sets are fine for n=64.
        common = adj[u] & adj[v]
        edge_density[key] = len(common)
        
    # Sort edges by density descending, then by degree sum descending as tie-breaker
    sorted_edges = sorted(edges, key=lambda e: (edge_density[_edge_key(e[0], e[1])], len(adj[e[0]]) + len(adj[e[1]])), reverse=True)
    
    covered_edges = set()
    cliques = []
    
    # We need to be careful: an edge might be "covered" by a previous clique.
    # We iterate through the sorted list. If an edge is already covered, skip it.
    # If not, we try to form a clique containing this edge.
    
    # To form the largest clique containing edge (u, v):
    # 1. Start with C = {u, v}.
    # 2. Candidates = N(u) ∩ N(v) (vertices connected to both u and v).
    # 3. Greedily add vertices from Candidates to C.
    #    Priority for adding vertex w:
    #    - w must be connected to ALL vertices currently in C.
    #    - Among valid candidates, pick the one that maximizes potential growth.
    #      Heuristic: Maximize |N(w) ∩ Candidates_remaining| or simply degree in the subgraph induced by C ∪ Candidates.
    
    for e in sorted_edges:
        if not _deadline_ok():
            break
            
        u, v = e
        key = _edge_key(u, v)
        
        if key in covered_edges:
            continue
            
        # Start clique with u, v
        clique = [u, v]
        covered_edges.add(key)
        
        # Find common neighbors of u and v that are not yet "used" in this clique construction?
        # Actually, we just need to find a clique in the original graph containing u and v.
        # The edges of this clique will be marked as covered.
        # We want to maximize the size of this clique to cover more edges.
        
        # Candidates: vertices w such that w is connected to u and v.
        # Note: w can be any vertex in the graph, even if its edges are already covered by other cliques?
        # No, if w's edges are covered, it doesn't matter for validity, but it doesn't help us cover NEW edges.
        # However, the clique must be a set of vertices. The edges within it must be in the graph.
        # If we add w, we cover edges (w, x) for all x in clique.
        # Some of those edges might already be covered. That's fine, we just don't "gain" anything for them.
        # But we gain the new edges.
        
        # So, we should pick w that maximizes the number of UNCOVERED edges (w, x) where x is in the current clique.
        
        # Initial state: Clique = {u, v}. Uncovered edges involving u or v?
        # We already marked (u, v) as covered.
        
        candidates = adj[u] & adj[v]
        # Remove u and v from candidates (they are in the clique)
        candidates.discard(u)
        candidates.discard(v)
        
        # Greedy expansion
        while candidates:
            if not _deadline_ok():
                break
                
            best_w = None
            best_score = -1
            
            # For each candidate w, check if it is connected to ALL vertices in current clique.
            # If yes, calculate score: number of edges (w, x) for x in clique that are NOT in covered_edges.
            
            for w in candidates:
                # Check connectivity to all in clique
                is_clique = True
                for x in clique:
                    if x not in adj[w]:
                        is_clique = False
                        break
                if not is_clique:
                    continue
                
                # Calculate potential gain
                gain = 0
                for x in clique:
                    ek = _edge_key(w, x)
                    if ek not in covered_edges:
                        gain += 1
                
                # Tie-breaker: prefer w with higher degree in the remaining candidate set?
                # Or just highest gain.
                if gain > best_score:
                    best_score = gain
                    best_w = w
            
            if best_w is None:
                break
                
            # Add best_w to clique
            clique.append(best_w)
            candidates.remove(best_w)
            
            # Mark all edges between best_w and existing clique members as covered
            for x in clique[:-1]: # x in clique before adding best_w
                ek = _edge_key(best_w, x)
                covered_edges.add(ek)
                
            # Update candidates: remove any vertex that is not connected to best_w
            # A vertex z in candidates must be connected to best_w to remain a candidate for the next step.
            candidates_to_remove = []
            for z in candidates:
                if best_w not in adj[z]:
                    candidates_to_remove.append(z)
            for z in candidates_to_remove:
                candidates.remove(z)
                
            # If gain was 0, we might still add it? 
            # If gain is 0, all edges (w, x) are already covered. 
            # Adding w doesn't help cover new edges. 
            # Should we stop? 
            # Yes, if gain is 0, we are not covering any new edges. 
            # However, maybe adding w allows adding y later which has high gain?
            # Unlikely in a dense graph if we pick high density edges first.
            # Let's stop if gain == 0 to avoid bloat.
            if best_score == 0:
                # Reconsider: if we don't add w, we might miss a larger clique.
                # But if all edges to current clique are covered, w is "redundant" for edge coverage.
                # Let's break if gain is 0.
                break

        # Add the clique to the result
        if len(clique) >= 2:
            cliques.append(clique)

    # Fallback: any remaining edges that were not covered (should be none if logic is correct, 
    # but just in case of timeout or logic gap)
    # Actually, the loop iterates through all edges. If an edge was not covered by a generated clique,
    # it means it was skipped because it was already covered? No, if it was covered, we skip it.
    # If it was NOT covered, we generate a clique for it.
    # So all edges should be covered.
    
    # However, if we timed out during the loop, some edges might be left uncovered.
    # We must ensure every edge is in exactly one clique.
    # If we broke early, we need to handle remaining edges.
    
    remaining_edges = all_edges - covered_edges
    for e in remaining_edges:
        if not _deadline_ok():
             # Emergency fallback: single edges
             cliques.append([e[0], e[1]])
             covered_edges.add(e)
        else:
             # Try to form a clique for this specific edge quickly
             u, v = e
             key = _edge_key(u, v)
             # Simple expansion: just take the edge and maybe one more if possible?
             # To be safe and fast, just add as single edge clique if no time.
             # But we have time in this branch.
             # Let's do a quick greedy expansion similar to above but limited.
             clique = [u, v]
             covered_edges.add(key)
             
             candidates = adj[u] & adj[v]
             candidates.discard(u)
             candidates.discard(v)
             
             # Limit expansion to a few steps to save time
             steps = 0
             while candidates and steps < 5:
                 best_w = None
                 best_gain = -1
                 for w in candidates:
                     is_clique = True
                     for x in clique:
                         if x not in adj[w]:
                             is_clique = False
                             break
                     if not is_clique:
                         continue
                     gain = 0
                     for x in clique:
                         ek = _edge_key(w, x)
                         if ek not in covered_edges:
                             gain += 1
                     if gain > best_gain:
                         best_gain = gain
                         best_w = w
                 
                 if best_w is None or best_gain == 0:
                     break
                 
                 clique.append(best_w)
                 candidates.remove(best_w)
                 for x in clique[:-1]:
                     covered_edges.add(_edge_key(best_w, x))
                 
                 # Update candidates
                 to_rm = [z for z in candidates if best_w not in adj[z]]
                 for z in to_rm:
                     candidates.remove(z)
                 steps += 1
                 
             cliques.append(clique)

    return cliques


def _fallback_single_edges(n, p, seed):
    edges = _random_edges(n, p, seed)
    return [[u, v] for u, v in edges]


def construct():
    """
    Returns [cliques_for_instance1, cliques_for_instance2].
    """
    res1 = _solve_instance(32, 0.6, 601)
    if not _deadline_ok():
        res1 = _fallback_single_edges(32, 0.6, 601)

    res2 = _solve_instance(64, 0.55, 602)
    if not _deadline_ok():
        res2 = _fallback_single_edges(64, 0.55, 602)

    return [res1, res2]
# EVOLVE-BLOCK-END
