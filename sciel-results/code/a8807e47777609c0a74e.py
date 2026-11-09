"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # Safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency bitmask
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    all_vertices_mask = (1 << n) - 1
    
    # Enumerate all maximal cliques using Bron-Kerbosch with bitmasks
    max_cliques = []
    
    def bron_kerbosch(R, P, X):
        if not _deadline_ok():
            return
        
        if P == 0 and X == 0:
            # R is a maximal clique
            if len(R) >= 2:
                # Store as a list of vertices
                vertices = [i for i in range(n) if (R >> i) & 1]
                max_cliques.append(vertices)
            return
        
        # Choose pivot u from P union X to minimize branching
        P_union_X = P | X
        if P_union_X:
            # Pick pivot with max degree in P (or just any, but max degree is better)
            # For simplicity and speed, pick the vertex with highest bit index?
            # Or iterate to find one with max (P & adj_mask[u])
            best_u = -1
            best_count = -1
            temp = P_union_X
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                count = bin(P & adj_mask[u]).count('1')
                if count > best_count:
                    best_count = count
                    best_u = u
                temp ^= lsb
            
            candidates = P & ~adj_mask[best_u]
        else:
            candidates = P
            
        temp = candidates
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            bron_kerbosch(R | lsb, P & adj_mask[u], X & adj_mask[u])
            P ^= lsb
            X |= lsb
            temp ^= lsb
            
    bron_kerbosch(0, all_vertices_mask, 0)
    
    if not _deadline_ok():
        # Fallback: return edges as cliques
        return [[u, v] for u, v in edges]
        
    # Sort maximal cliques by size descending (larger cliques cover more edges)
    # This is a heuristic for greedy set packing
    max_cliques.sort(key=lambda c: len(c), reverse=True)
    
    # Map clique to its edge bitmask for fast intersection
    # Edge (u, v) where u < v is represented by bit index u * n + v? 
    # No, n is up to 64, so n*n = 4096 bits. Python ints handle this easily.
    # Or we can just use a set of frozensets of edges?
    # Bitmask is faster for intersection.
    # Let's use a dictionary mapping clique index to edge bitmask.
    
    clique_edge_masks = []
    for clique in max_cliques:
        mask = 0
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u = clique[i]
                v = clique[j]
                if u > v:
                    u, v = v, u
                # Index: u * n + v is unique for u < v
                # Max index: 63 * 64 + 63 = 4105. Fits in Python int.
                mask |= (1 << (u * n + v))
        clique_edge_masks.append(mask)
        
    remaining_edges_mask = 0
    for u, v in edges:
        if u > v:
            u, v = v, u
        remaining_edges_mask |= (1 << (u * n + v))
        
    partition = []
    
    # Greedy set packing
    # We want to pick cliques that cover the most remaining edges.
    # Since we sorted by size initially, we can just iterate?
    # No, a smaller clique might cover more *remaining* edges if larger ones overlap.
    # But for a first pass, iterating sorted by size is a good heuristic.
    # To improve, we could re-evaluate "coverage" dynamically, but that's O(M^2).
    # Let's try a simple greedy: iterate through sorted cliques.
    # If a clique covers any remaining edges, take it.
    
    # Optimization: 
    # If we just iterate, we might take a large clique that shares edges with many others,
    # preventing them from being taken.
    # However, since we remove covered edges, subsequent cliques will only cover what's left.
    # The "greedy maximal clique cover" usually works well.
    
    # Let's add a bit of logic: 
    # If a clique's intersection with remaining edges is small relative to its size, maybe skip?
    # No, any edge covered is good.
    
    # One improvement: if a clique is a subset of another clique, the larger one dominates.
    # But maximal cliques are not subsets of each other.
    
    # Let's just iterate.
    for i, clique in enumerate(max_cliques):
        if not _deadline_ok():
            break
            
        ce_mask = clique_edge_masks[i]
        covered_edges = ce_mask & remaining_edges_mask
        
        if covered_edges:
            # Take this clique
            partition.append(clique)
            remaining_edges_mask ^= covered_edges # Remove covered edges
            
            if not remaining_edges_mask:
                break
                
    # If there are remaining edges, they are not covered by any maximal clique?
    # That's impossible because every edge is in some maximal clique.
    # Wait, if we skipped a clique because it had 0 intersection with *remaining* edges,
    # that's fine.
    # But if we finish the loop and remaining_edges_mask is not 0,
    # it means some edges were not covered by any clique we picked.
    # This can happen if we picked a clique that "blocked" others?
    # No, we only remove edges that are covered.
    # If an edge is in remaining, it must be in some maximal clique.
    # Did we skip that maximal clique?
    # We only skip if `covered_edges` is 0.
    # If `covered_edges` is 0, it means none of its edges are in `remaining`.
    # So if an edge is in `remaining`, the maximal clique containing it MUST have had
    # at least that edge in `remaining` when we checked it, UNLESS that edge was removed
    # by a PREVIOUS clique.
    # If the edge was removed by a previous clique, it wouldn't be in `remaining` now.
    # So if `remaining` is not 0, it implies a bug in logic or we ran out of time.
    # Actually, if we run out of time, we break.
    # If we finish the loop, `remaining` should be 0.
    
    # Fallback for any leftover edges (e.g. if deadline hit or logic gap)
    if remaining_edges_mask:
        # Convert remaining bitmask back to edges
        temp = remaining_edges_mask
        while temp:
            if not _deadline_ok():
                break
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            u = idx // n
            v = idx % n
            partition.append([u, v])
            temp ^= lsb
            
    return partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if not _deadline_ok():
            # If we run out of time before starting an instance, return empty/invalid?
            # Better to try to solve as much as possible.
            # But if we can't even start, we might return [] which is invalid if edges exist.
            # Let's just try to solve.
            pass
        
        # Solve instance
        res = _solve_instance(n, p, seed)
        results.append(res)
        
    return results
# EVOLVE-BLOCK-END
