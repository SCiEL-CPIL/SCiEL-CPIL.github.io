"""
Ramsey graphs for R(5,5): large graphs with no K5 and no independent 5-set.
Constructs the Paley graph on 41 vertices (if valid) or 37 vertices (fallback).
"""

import time
import random
from itertools import combinations


def _has_clique(adj, n, k):
    """
    Check if the graph (given by adjacency bitmasks) has a clique of size k.
    Uses backtracking with bitmasks.
    """
    if k == 1:
        return True
    if k == 0:
        return True

    # We want to find a clique of size k in the full graph.
    # Start with all vertices as candidates.
    full_mask = (1 << n) - 1

    def _backtrack(candidates, remaining):
        if remaining == 1:
            return candidates != 0
        if candidates == 0:
            return False

        # Pick the lowest set bit (vertex)
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1

        # Option 1: Include v
        # The new candidates must be neighbors of v that are in the current candidates
        # and have index > v (to avoid duplicates? Actually, bitmask approach handles order naturally if we just intersect).
        # Wait, standard clique finding:
        # If we include v, we must find a clique of size 'remaining-1' in the intersection of 'candidates' and 'adj[v]'.
        # To ensure we don't count the same clique multiple times or get stuck in infinite loops,
        # we should restrict candidates to vertices with index > v?
        # No, the bitmask intersection `adj[v] & candidates` is sufficient if we remove v from candidates before recursing?
        # Actually, if we include v, the next vertices must be neighbors of v.
        # So candidates_new = candidates & adj[v] & ~(1 << v)  # Remove v itself
        # But since adj[v] doesn't have bit v set (no self-loops), candidates & adj[v] is fine.
        
        nbrs = candidates & adj[v]
        if _backtrack(nbrs, remaining - 1):
            return True

        # Option 2: Exclude v
        candidates_without_v = candidates & ~lsb
        if _backtrack(candidates_without_v, remaining):
            return True

        return False

    return _backtrack(full_mask, k)


def _has_independent_set(adj, n, k):
    """
    Check if the graph has an independent set of size k.
    An independent set in G is a clique in the complement of G.
    """
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        # Complement adjacency: all bits set except self and neighbors in G
        comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
    
    return _has_clique(comp_adj, n, k)


def _verify_55_free(adj, n, cliquesize=5, indsize=5):
    """
    Verify that the graph has no clique of size cliquesize and no independent set of size indsize.
    """
    if _has_clique(adj, n, cliquesize):
        return False
    if _has_independent_set(adj, n, indsize):
        return False
    return True


def _build_adjacency(n, edges):
    """Build adjacency list as bitmasks from edge list."""
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj


def _paley_graph(q):
    """
    Construct the Paley graph on q vertices.
    q must be a prime power congruent to 1 mod 4.
    Vertices are 0..q-1. Edge (i, j) if (j-i) mod q is a quadratic residue.
    """
    if q % 4 != 1:
        raise ValueError("q must be 1 mod 4 for Paley graph")
    
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges


def _clebsch_graph_complement():
    """
    Return the complement of the Clebsch graph (16 vertices).
    The Clebsch graph is the folded 5-cube. It is strongly regular (16, 5, 0, 2).
    Its complement is strongly regular (16, 10, 6, 6).
    This is a (16, 5, 5)-free graph? 
    Actually, the Clebsch graph itself is (16, 5, 5)-free? 
    The independence number of the Clebsch graph is 5. So it has an independent set of size 5.
    The clique number of the Clebsch graph is 3 (triangle-free). So no K5.
    But it has an independent set of size 5. So it is NOT (5,5)-free.
    
    What about the complement?
    The complement of the Clebsch graph has clique number = independence number of Clebsch = 5.
    So the complement has a K5.
    So neither the Clebsch graph nor its complement is (5,5)-free.
    
    However, the prompt suggests using the complement of the Clebsch graph as a base for expansion.
    Maybe the idea is to use it as a starting point for a local search or duplication heuristic?
    But for now, let's stick to the known valid Paley graphs.
    """
    # Construct Clebsch graph edges
    # Clebsch graph vertices can be represented as binary strings of length 5 with even parity.
    # There are 16 such strings.
    # Two vertices are adjacent if they differ in exactly 3 bits? 
    # Wait, the Clebsch graph is the folded 5-cube.
    # Vertices are binary strings of length 5 with even weight.
    # Two vertices are adjacent if their Hamming distance is 3.
    
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
    
    # Map index to vertex value
    idx_to_val = {i: v for i, v in enumerate(vertices)}
    val_to_idx = {v: i for i, v in enumerate(vertices)}
    
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            vi = vertices[i]
            vj = vertices[j]
            # Hamming distance
            dist = bin(vi ^ vj).count('1')
            if dist == 3:
                edges.append((i, j))
    
    # This is the Clebsch graph.
    # Return the complement?
    # The prompt says "Use the complement of the Clebsch graph (16 vertices) as a base".
    # Let's return the complement edges.
    all_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    clebsch_set = set(edges)
    comp_edges = [e for e in all_edges if e not in clebsch_set]
    return 16, comp_edges


def construct():
    """
    Construct a graph with no K5 and no independent 5-set.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 300  # Safety margin

    # Try Paley(41) first
    try:
        n_41, edges_41 = _paley_graph(41)
        adj_41 = _build_adjacency(n_41, edges_41)
        if _verify_55_free(adj_41, n_41):
            return n_41, edges_41
    except Exception:
        pass

    # Fallback to Paley(37)
    try:
        n_37, edges_37 = _paley_graph(37)
        adj_37 = _build_adjacency(n_37, edges_37)
        if _verify_55_free(adj_37, n_37):
            return n_37, edges_37
    except Exception:
        pass

    # If all else fails, return an empty graph (n=0)
    return 0, []


# EVOLVE-BLOCK-END
