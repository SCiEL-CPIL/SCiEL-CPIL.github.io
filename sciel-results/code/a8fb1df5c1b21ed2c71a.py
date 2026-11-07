"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    """
    Strategy: Direct constructive design (Ladder Augmentation).
    
    We add edges to create a 'ladder' structure on the path graph.
    By adding edges (i, i+4) for i in 0..19, we create short-cuts.
    The graph remains bipartite (triangle-free) because all added edges connect nodes of different parity (distance 4 is even).
    The diameter is reduced to 4.
    
    Let's verify the diameter for the ladder on 0..23 with edges (i, i+1) and (i, i+4).
    Max distance is between 0 and 23.
    Path: 0 -> 4 -> 8 -> 12 -> 16 -> 20 -> 23 (length 6). Wait, diameter must be <= 4.
    
    Let's re-evaluate.
    If we add (i, i+4), the "step" is 4.
    Distance from 0 to 23:
    0-4-8-12-16-20-23 is 6 steps.
    We need diameter <= 4.
    
    Let's try adding edges (i, i+5).
    Parity: 5 is odd, so it preserves bipartiteness.
    Distance 0 to 23:
    0 -> 5 -> 10 -> 15 -> 20 -> 23 (5 steps). Still > 4.
    
    Let's try a mix or larger jumps.
    To get diameter 4 on 24 nodes, we need very fast traversal.
    Consider a "hub" strategy but constrained to be triangle-free.
    A star graph has diameter 2 but contains triangles if the center connects to nodes that are connected? No, a star is triangle-free.
    But we must keep the path.
    If we add a center node, say 12, and connect it to everyone?
    Edges (12, v) for all v.
    Triangle check:
    Path has (11, 12) and (12, 13).
    If we add (12, 11), it's a duplicate.
    If we add (12, 13), duplicate.
    If we add (12, 10). Triangle? (10, 11, 12). Edge (10, 11) exists, (11, 12) exists, (12, 10) added. Triangle!
    So we cannot connect a node to its 2-neighbor.
    
    Triangle-free constraint on path P_n:
    We can add edge (u, v) only if |u-v| is not 2 (would form triangle with u-1 or u+1? No, (u, u+2) forms triangle with u+1).
    Actually, if we add (u, v), we must ensure no w such that (u, w) and (w, v) are in the graph.
    Since the path is there, (u, u+1) and (u+1, u+2) exist. So we cannot add (u, u+2).
    Also, if we add multiple edges, we must check for new triangles formed by the new edges.
    
    Let's look at the "single hub" finding: "single hub connected to all vertices in the opposite bipartition".
    Bipartition of P_24:
    Even: 0, 2, ..., 22
    Odd: 1, 3, ..., 23
    
    If we pick a hub, say 0 (Even).
    Connect 0 to all Odd nodes: 1, 3, 5, ..., 23.
    Edge (0, 1) is already in path.
    Add (0, 3), (0, 5), ..., (0, 23).
    Is it triangle-free?
    Any triangle involving 0 must use two neighbors of 0.
    Neighbors of 0 in new graph: 1, 3, 5, ..., 23.
    Are any of these connected to each other?
    In the path, 1 is connected to 2. 3 to 2 and 4.
    Is 1 connected to 3? No (distance 2).
    Is 3 connected to 5? No.
    Are any two odd nodes connected in the path? No, path is bipartite.
    Are any two odd nodes connected via other added edges? No, only 0 is connected to them.
    So no triangle includes 0.
    What about triangles not including 0?
    The original path is triangle-free.
    The added edges are all incident to 0.
    So no new triangles are formed.
    
    Is it connected? Yes, 0 is connected to all odd nodes, and odd nodes are connected to even nodes via the path.
    
    What is the diameter?
    Max distance between any two nodes.
    0 is connected to all Odd nodes (dist 1).
    0 is connected to 1 (dist 1).
    Distance from 0 to any Even node v (v != 0):
    0 -> (some odd neighbor of v) -> v.
    Since v is even, v-1 and v+1 are odd.
    0 is connected to v-1 and v+1 (if they exist and are odd).
    So dist(0, v) = 2.
    
    Distance between two Even nodes u, v (u, v != 0):
    u -> (odd neighbor) -> 0 -> (odd neighbor of v) -> v.
    Length 3.
    Can it be 2? u -> w -> v. w must be common neighbor.
    If w is odd, it must be connected to u and v in the path.
    u and v are even.
    If |u-v|=2, they share an odd neighbor. Dist 2.
    If |u-v|>2, no common odd neighbor in path.
    Do they share a common neighbor via 0?
    u -> 0 -> v? No, 0 is not connected to u (u is even, 0 is even, no edge).
    So dist(u, v) = 3 if |u-v| > 2.
    
    Distance between two Odd nodes u, v:
    u -> 0 -> v. Length 2.
    
    Distance between 0 and Odd node: 1.
    Distance between 0 and Even node: 2.
    
    So the diameter is max(2, 3, 2) = 3.
    This satisfies diameter <= 4.
    
    Number of added edges:
    Odd nodes: 1, 3, ..., 23.
    Total 12 odd nodes.
    Edge (0, 1) exists.
    Add (0, 3), (0, 5), ..., (0, 23).
    Count: 11 edges.
    
    This matches the "cell finding" of 11 edges.
    The current program adds ~121 edges.
    Score = Seed / Value.
    Seed value (num edges) is ~121.
    New value is 11.
    Score = 121 / 11 = 11.
    This is a massive improvement over 1.0.
    
    Implementation:
    Base edges are path.
    Added edges: [(0, v) for v in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]]
    """
    added = []
    # Hub 0 (even) connects to all odd nodes except 1 (already connected)
    for v in range(3, 24, 2):
        added.append([0, v])
    
    # Verify types
    result = []
    for e in added:
        result.append([int(e[0]), int(e[1])])
    return result
# EVOLVE-BLOCK-END
