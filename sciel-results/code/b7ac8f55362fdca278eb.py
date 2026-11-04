import time
import random
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Petersen graph edges
    # Vertices 0-9
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def build_cover(m, voltages):
        # voltages: dict {(u, v): val}
        n = 10 * m
        edges = []
        for u in range(10):
            for v in range(m):
                for (a, b) in petersen_edges:
                    if a == u:
                        vol = voltages.get((a, b), voltages.get((b, a), 0))
                        # Ensure direction consistent
                        if a > b:
                            vol = -vol % m
                        w = (b, (v + vol) % m)
                        i = u * m + v
                        j = w[0] * m + w[1]
                        if i < j:
                            edges.append((i, j))
        return n, edges

    def check_girth(n, edges, target_girth):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles < target_girth
        # BFS from each node? Too slow for n=640.
        # Use non-backtracking walk check.
        # If no non-backtracking word of length < g is identity, girth >= g.
        
        # We can check by BFS on the state (node, prev_node).
        # Start from each node?
        # Actually, just check if any cycle of length < target_girth exists.
        
        # For n=640, we can just do BFS from each node up to depth target_girth/2?
        # No, just check all nodes.
        
        # Optimization: If we find a cycle, return False.
        # We can use a set of visited (node, prev) for each start?
        # Better: Just generate all non-backtracking walks of length L < target_girth.
        # If any start and end at the same node (and L > 2), it's a cycle.
        
        # Actually, simpler:
        # For each node u, do a BFS up to depth target_girth - 1.
        # If we reach u again (with a valid non-backtracking path), we found a cycle.
        
        # This is O(n * (k-1)^(g/2)).
        # 640 * 2^7 = 640 * 128 = 81920 operations. Very fast.
        
        for u in range(n):
            # BFS from u
            # State: (current, prev)
            # Start: (u, -1)
            # Queue: [(u, -1, 0)]
            visited = set()
            visited.add((u, -1))
            queue = [(u, -1, 0)]
            
            while queue:
                curr, prev, dist = queue.pop(0)
                if dist >= target_girth:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == u and dist > 0:
                        # Found a cycle of length dist + 1
                        # Ensure it's a simple cycle?
                        # BFS ensures shortest path.
                        # If we hit u, the path is a cycle.
                        return False
                    state = (nxt, curr)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nxt, curr, dist + 1))
        return True

    # Try voltage lifts
    for m in [64, 128, 256]:
        if time.time() > deadline - 10:
            break
            
        # Try random voltage assignments
        found = False
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            voltages = {}
            for u, v in petersen_edges:
                voltages[(u, v)] = random.randint(0, m - 1)
            
            n, edges = build_cover(m, voltages)
            
            # Quick check: girth
            if check_girth(n, edges, g):
                return n, edges
    return None
