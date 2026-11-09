"""Construct a dense critical diameter-2 graph on 16 vertices."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return edges of the complete bipartite graph K_{8,8} on vertices 0..15.
    
    Properties:
    - Diameter is exactly 2.
    - Every edge is critical: deleting any edge disconnects the specific pair 
      into a distance-3 path (since no other short paths exist in a complete 
      bipartite graph between the specific endpoints if the direct edge is gone, 
      wait, in K_{8,8} removing (u,v) makes dist(u,v)=3, so diameter becomes 3).
    - Edge count: 64.
    - This is the maximum number of edges for a minimal diameter-2 graph 
      (Turán's theorem bound for triangle-free, and minimal diameter-2 
      graphs are extremal at n^2/4).
    """
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    return edges

# EVOLVE-BLOCK-END
