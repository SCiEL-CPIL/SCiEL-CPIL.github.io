import numpy as np
import time
import random
import itertools

def count_c4(adj):
    """Count 4-cycles in a graph represented by adjacency matrix."""
    n = len(adj)
    total = 0
    for u in range(n):
        for v in range(u+1, n):
            c = np.sum(adj[u] & adj[v])
            if c >= 2:
                total += c * (c - 1) // 2
    return total // 2

def count_c4_incremental(adj, u, v):
    """Count how many 4-cycles are created by adding edge (u,v).
    A 4-cycle is created for each pair of common neighbors w1, w2 of u and v
    such that w1-w2 is NOT an edge (otherwise it's a triangle, not a C4).
    Actually, a C4 containing edge (u,v) has form u-w1-x-w2-v-u where w1,w2 are
    common neighbors of u and v, and x is a common neighbor of w1 and w2.
    More precisely: adding edge (u,v) creates a C4 for each pair (w1,w2) where
    w1,w2 are in N(u)∩N(v), and there exists x in N(w1)∩N(w2) with x != u,v.
    But simpler: each 4-cycle containing (u,v) corresponds to a path u-w1-x-w2-v
    of length 3 from u to v. The number of such paths equals the number of
    length-3 paths from u to v in the graph without edge (u,v).
    
    A length-3 path u-a-b-v exists iff a is neighbor of u, b is neighbor of v,
    and a-b is an edge. So count = sum over a in N(u) of |N(a) ∩ N(v) \ {u}|.
    But we need to be careful not to double count. Actually each C4 containing
    edge (u,v) is counted exactly once this way if we fix the "opposite" edge.
    
    Let me think again. A C4 is u-v-b-a-u (or u-v-a-b-u). If we add edge (u,v),
    then for each pair (a,b) where a~u, b~v, and a~b, we get a 4-cycle u-v-b-a-u.
    But we must ensure it's a simple cycle: a != v, b != u, a != b.
    Since a~u and a != u (no self-loops), and b~v and b != v.
    Also need a != b for it to be a 4-cycle (not a triangle with a duplicate).
    Actually if a == b, then u-a-v is a triangle, and u-v-a-u is not a 4-cycle.
    So we need a != b.
    
    So the number of new C4s = number of pairs (a,b) with a in N(u), b in N(v),
    a~b, and a != b.
    """
    # Get neighbors of u and v
    nu = np.where(adj[u])[0]
    nv = np.where(adj[v])[0]
    
    count = 0
    for a in nu:
        if a == v:
            continue
        for b in nv:
            if b == u or b == a:
                continue
            if adj[a, b]:
                count += 1
    return count

def construct():
    n = 31
    target_edges = 91
    deadline = time.time() + 110
    
    best_c4 = float('inf')
    best_edges = None
    
    # Try multiple random initial orderings with greedy edge selection
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        adj = np.zeros((n, n), dtype=bool)
        edges = set()
        
        all_pairs = list(itertools.combinations(range(n), 2))
        random.shuffle(all_pairs)
        
        # Greedy: add edges one by one, choosing the one that minimizes C4 count
        # We'll use a priority-based approach: for each candidate edge, compute
        # the incremental C4 cost, and pick the cheapest.
        
        # For efficiency, we'll do a limited lookahead greedy
        for _ in range(target_edges):
            if time.time() > deadline - 5:
                break
            
            # Compute cost for each non-edge
            best_edge = None
            best_cost = float('inf')
            
            # Sample a subset of non-edges for speed
            non_edges = [p for p in all_pairs if p not in edges]
            if not non_edges:
                break
            
            # Use a heuristic: prefer edges between low-degree vertices
            # and check a random subset for exact cost
            candidates = non_edges[:200]
            random.shuffle(candidates)
            
            for u, v in candidates:
                cost = count_c4_incremental(adj, u, v)
                # Tie-break by degree to keep graph balanced
                deg_u = np.sum(adj[u])
                deg_v = np.sum(adj[v])
                total_cost = cost * 10 + deg_u + deg_v
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_edge = (u, v)
            
            if best_edge is None:
                best_edge = random.choice(non_edges)
            
            u, v = best_edge
            adj[u, v] = adj[v, u] = True
            edges.add(best_edge)
        
        c4 = count_c4(adj)
        if c4 < best_c4:
            best_c4 = c4
            best_edges = set(edges)
    
    if best_edges is None:
        # Fallback: just return any 91 edges
        all_pairs = list(itertools.combinations(range(n), 2))
        random.seed(42)
        random.shuffle(all_pairs)
        best_edges = set(all_pairs[:target_edges])
    
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
# EVOLVE-BLOCK-END
