"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    """
    Strategy: Independent Dominating Set (IDS) based approach.
    
    We model the problem as finding a minimum subset S of vertices such that:
    1. S is an independent set (no two vertices in S are adjacent).
    2. S is a dominating set (every vertex is in S or adjacent to a vertex in S).
    
    For a path graph P_n, the minimum IDS size is ceil(n/3).
    For n=24, this is 8 vertices.
    
    However, we also need diameter <= 4 and triangle-free.
    The path itself is triangle-free. If we add edges from a hub to vertices in S,
    we must ensure no triangles are formed.
    
    A simpler approach: Use a hub strategy but optimize the hub choice and connections.
    The current best is 11 edges (hub 0 connected to all odd nodes except 1).
    
    Let's try a different structure: Two hubs.
    Hub 1 (odd) and Hub 22 (even).
    Connect Hub 1 to all even nodes except its neighbors (0, 2).
    Connect Hub 22 to all odd nodes except its neighbors (21, 23).
    
    Let's verify triangle-freeness and diameter.
    """
    
    # Fallback to known good solution if optimization fails or takes too long
    def fallback():
        added = []
        for v in range(3, 24, 2):
            added.append([0, v])
        return [[int(e[0]), int(e[1])] for e in added]
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Try to find a better solution using a simple local search or exhaustive check for small cases
    # Since n=24 is small, we can try a few structural patterns.
    
    # Pattern 1: Single Hub (Current Best: 11 edges)
    # Hub 0 connects to 3,5,7,9,11,13,15,17,19,21,23 (11 edges)
    
    # Pattern 2: Two Hubs
    # Try Hub 1 and Hub 22
    # Hub 1 (odd) connects to even nodes not adjacent to 1: 4,6,8,10,12,14,16,18,20 (9 edges)
    # Hub 22 (even) connects to odd nodes not adjacent to 22: 1,3,5,7,9,11,13,15,17,19 (10 edges)
    # Total 19 edges. Worse.
    
    # Pattern 3: One Hub, but maybe a different hub?
    # Hub 11 (odd). Connect to even nodes not adjacent to 11: 0,2,4,6,8,14,16,18,20,22 (10 edges).
    # This is better than 11!
    
    added = []
    hub = 11
    # Hub 11 is odd. Neighbors in path: 10, 12.
    # Connect to all even nodes except 10 and 12.
    for v in range(0, 24, 2):
        if v == 10 or v == 12:
            continue
        if v == hub:
            continue
        added.append([hub, v])
        
    # Check triangle-freeness for this specific pattern
    # Hub 11 connects to 0,2,4,6,8,14,16,18,20,22.
    # Are any of these adjacent to each other? No, all even.
    # Are any of these adjacent to 11 in the path? 10 and 12 are, but we excluded them.
    # So no triangle includes 11.
    # No other edges added. So triangle-free.
    
    # Check diameter
    # Dist(11, v) = 1 for v in {0,2,4,6,8,14,16,18,20,22}.
    # Dist(11, 10) = 1 (path).
    # Dist(11, 12) = 1 (path).
    # Dist(11, odd nodes):
    #   11 -> 10 -> 9 (2)
    #   11 -> 12 -> 13 (2)
    #   11 -> 10 -> 8 -> 7 (3) ... wait
    #   Let's compute max dist.
    #   Farthest odd node from 11? 1 or 23.
    #   Dist(11, 1): 11->10->9->8->7->6->5->4->3->2->1 (10 steps) OR 11->0->1 (2 steps).
    #   Since 11-0 is an added edge, dist(11,1) = 2.
    #   Dist(11, 23): 11->22->23 (2 steps).
    #   Dist(11, 3): 11->2->3 (2 steps).
    #   Dist(11, 21): 11->22->21 (2 steps).
    #   So all odd nodes are within distance 2 of 11.
    
    # Dist between two even nodes u, v (not 11):
    #   u -> 11 -> v. Length 2.
    # Dist between two odd nodes u, v:
    #   u -> (even neighbor) -> 11 -> (even neighbor of v) -> v. Length 4.
    #   Can it be 3? u -> w -> x -> v.
    #   If u and v share a common even neighbor, dist 2.
    #   If not, dist 4 via hub.
    #   Max dist is 4.
    
    # Number of edges: 10.
    
    if time.time() > deadline:
        return fallback()
        
    # Verify types
    result = []
    for e in added:
        result.append([int(e[0]), int(e[1])])
    return result
# EVOLVE-BLOCK-END
