import itertools
import time
import random

def _q3_edge_sets():
    """Generate all edge sets of Q3 embeddings on 16 vertices."""
    # Q3 vertices are 3-bit tuples
    q3_vertices = list(itertools.product([0, 1], repeat=3))
    # Q3 edges: pairs differing in exactly one coordinate
    q3_edges = set()
    for i, u in enumerate(q3_vertices):
        for j, v in enumerate(q3_vertices):
            if i < j:
                diff = sum(1 for a, b in zip(u, v) if a != b)
                if diff == 1:
                    q3_edges.add((i, j))
    
    # We need to find all injective mappings from Q3 vertices to {0..15}
    # such that Q3 edges map to graph edges
    # This is expensive, so we use a smarter approach
    
    # Actually, for n=16, the number of Q3 embeddings is manageable
    # Let's enumerate all permutations of 8 vertices from 16 and check
    # if they form a Q3
    
    # Better approach: generate all 8-subsets and check if they induce a Q3
    # But that's C(16,8) = 12870, and for each we need to check all 8! labelings
    # That's too slow.
    
    # Alternative: use the fact that Q3 is the Cartesian product of three K2s
    # A Q3 embedding is determined by choosing three disjoint pairs of vertices
    # (the three "dimensions") such that the 8 vertices form a cube
    
    # Let's think differently: enumerate all ways to choose 8 vertices
    # and check if the induced subgraph is isomorphic to Q3
    
    # Actually, let's just hardcode the Q3 edge sets by generating them
    # from the standard Q3 structure
    
    # The Q3 has 8 vertices and 12 edges
    # We'll generate all edge sets by permuting vertex labels
    
    # For efficiency, we'll generate Q3 embeddings on the fly during the search
    pass

def _has_q3(edges_set, vertices):
    """Check if the graph with edges edges_set on vertices contains a Q3."""
    # For each 8-subset of vertices, check if it induces a Q3
    # This is expensive, so we use a smarter check
    
    # Build adjacency
    adj = {v: set() for v in vertices}
    for u, v in edges_set:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    
    # A Q3 is a 3-cube: 8 vertices, each degree 3, 12 edges
    # Key property: in Q3, each vertex has 3 neighbors, and the neighbors
    # of any vertex are pairwise non-adjacent (independent set)
    
    # Try all 8-subsets
    for subset in itertools.combinations(vertices, 8):
        sset = set(subset)
        # Check degrees within the subset
        degrees = {}
        for v in subset:
            deg = sum(1 for u in adj[v] if u in sset)
            degrees[v] = deg
            if deg != 3:
                break
        else:
            # All degrees are 3
            # Check edge count
            edge_count = sum(degrees[v] for v in subset) // 2
            if edge_count != 12:
                continue
            # Check that for each vertex, its 3 neighbors are pairwise non-adjacent
            valid = True
            for v in subset:
                neighbors = [u for u in adj[v] if u in sset]
                for i in range(3):
                    for j in range(i + 1, 3):
                        if neighbors[i] in adj[neighbors[j]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    vertices = list(range(16))
    all_edges = list(itertools.combinations(vertices, 2))
    
    # Start with K_{3,13}
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    best_edges = list(edges)
    
    # Greedy edge addition
    # Try to add edges in a random order
    remaining = [e for e in all_edges if e not in edges]
    random.shuffle(remaining)
    
    for u, v in remaining:
        if time.time() > deadline:
            break
        new_edges = edges | {(u, v)}
        if not _has_q3(new_edges, vertices):
            edges = new_edges
            best_edges = list(edges)
    
    # Local search: try to add more edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        remaining = [e for e in all_edges if e not in edges]
        random.shuffle(remaining)
        for u, v in remaining:
            if time.time() > deadline:
                break
            new_edges = edges | {(u, v)}
            if not _has_q3(new_edges, vertices):
                edges = new_edges
                best_edges = list(edges)
                improved = True
    
    # Convert to list of [u, v] pairs
    return [[u, v] for u, v in best_edges]
