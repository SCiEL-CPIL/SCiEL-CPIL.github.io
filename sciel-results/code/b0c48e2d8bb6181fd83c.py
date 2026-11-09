"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    # We build a C4-free graph on 20 vertices (to maximize edges subject to no 4-cycles)
    # and then assign colours so that any 4-cycle (if it existed, but we ensure none)
    # would be rainbow. Since C4-free graphs have no 4-cycles, the condition is vacuously true.
    # We use a greedy algorithm to build a dense C4-free graph.
    
    n = 20
    edges = []
    # adj[v] = set of neighbours
    adj = [set() for _ in range(n)]
    # edge_count[v] = degree
    # We will try to add edges in a way that avoids creating C4s.
    
    # A 4-cycle v1-v2-v3-v4-v1 is created if we add edge (v3,v4) and there exists v2 such that 
    # (v1,v2) and (v2,v4) exist and (v1,v3) exists? No, that's not right.
    # A 4-cycle is v1-v2-v3-v4-v1. It is formed if we add (v1,v4) and there is a path of length 3 
    # from v1 to v4, i.e., v1-u-w-v4 where (v1,u), (u,w), (w,v4) are edges.
    # Actually, a simpler check: when adding (u,v), a C4 is formed if there is a vertex w such that 
    # there is a path u-x-w-v or u-w-x-v. That is, if there exist x such that (u,x) and (x,v) exist? No, that's a path of length 2.
    # A C4 involving new edge (u,v) would be u-v-w-x-u or u-v-x-w-u.
    # This requires that there is a vertex w adjacent to v and a vertex x adjacent to u such that (w,x) is an edge.
    # So, when adding (u,v), we must ensure that there is no edge (w,x) where w is a neighbour of v and x is a neighbour of u.
    # Equivalently, the sets of neighbours of u and v (excluding each other) must not have any edge between them.
    
    # To make this efficient, we can maintain for each pair of vertices (u,v) whether an edge exists.
    # But n=20 is small, so we can just check all pairs.
    
    # We will try all pairs (u,v) in a random order and add the edge if it doesn't create a C4.
    # We shuffle the order to get different results, but for determinism we can use a fixed seed or a specific order.
    # Let's use a fixed random seed for reproducibility in evaluation.
    
    random.seed(42)
    candidates = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(candidates)
    
    for u, v in candidates:
        if u in adj[v]:
            continue
        # Check if adding (u,v) creates a C4
        # We need to check if there exist w in adj[v] and x in adj[u] such that (w,x) is an edge.
        # Note: w != u and x != v, but since adj sets don't contain self-loops, this is fine.
        # Also, we must ensure w != x, but that's implied by no self-loops.
        conflict = False
        for w in adj[v]:
            for x in adj[u]:
                if x in adj[w]:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    # Now we have a C4-free graph. We need to assign colours.
    # Since there are no 4-cycles, any colouring works. But we must use colours 0..19.
    # We can just assign colour 0 to all edges, or distribute them.
    # The task says "edge-coloured with 20 colours", but it doesn't say each colour must be used.
    # However, to be safe, let's assign distinct colours to edges if possible, or just use colour 0.
    # Actually, the condition is "every 4-cycle is rainbow". If there are no 4-cycles, it's satisfied.
    # So we can assign any colours. Let's assign colour 0 to all edges for simplicity.
    # But wait, the problem says "coloured with 20 colours". This might imply we have a palette of 20 colours, 
    # not that we must use all of them. Let's re-read: "edge-coloured with 20 colours so every 4-cycle is rainbow".
    # This usually means the colours come from a set of size 20. It doesn't require all to be used.
    # So assigning all edges colour 0 is valid.
    
    coloured_edges = [[u, v, 0] for u, v in edges]
    
    # We can try to improve the edge count by restarting with different seeds if the count is low.
    # But for now, let's see how many edges we get.
    # A random greedy C4-free graph on 20 vertices might have around 20-30 edges.
    # The maximum is known to be around 2n^(3/2) ~ 90 for large n, but for n=20, it's less.
    # Actually, the extremal number ex(n, C4) is O(n^(3/2)). For n=20, n^(3/2) = 89. But constants matter.
    # Let's just return the result.
    
    return coloured_edges
# EVOLVE-BLOCK-END
