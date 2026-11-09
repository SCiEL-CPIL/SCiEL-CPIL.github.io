import time
from itertools import combinations, product
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = 12
    min_deg = 3
    
    # Try to find a graph where all cycles have the same length
    # Strategy: try 3-regular bipartite graphs on 12 vertices
    
    # First, try a known good construction: two K_{3,3} gives 2 cycle lengths (4,6) -> score 2.5
    # Can we do better?
    
    # Let's try to find a graph with only one cycle length.
    # A 4-regular graph on 12 vertices where all cycles are length 4?
    
    # Actually, let's try the complement approach or other constructions.
    
    # Key idea: If we can find a graph where all simple cycles have the same length,
    # the raw value is 1, giving score = seed/1 = 2.0 (if seed=2.0) or better.
    
    # Let's try: a graph that is a "thickened" cycle. 
    # For example, take a 6-cycle and add parallel paths to make it 3-regular.
    
    # Alternative: Use a bipartite graph with parts of size 6, and ensure
    # all cycles are length 4.
    
    # A bipartite graph where every cycle is length 4 is one where no two 
    # vertices in the same part have two common neighbors (that would create 
    # multiple 4-cycles but that's fine) AND no cycles of length 6+.
    
    # Actually, a bipartite graph has all cycles of even length. To have only 
    # 4-cycles, we need no 6-cycles, 8-cycles, etc.
    
    # Let me try a specific construction:
    # Part A: {0,1,2,3,4,5}, Part B: {6,7,8,9,10,11}
    # Each vertex in A connects to 3 vertices in B, and vice versa.
    # We want to avoid 6-cycles.
    
    # A 6-cycle in bipartite graph: a1-b1-a2-b2-a3-b3-a1
    # This requires a1,b1,a2,b2,a3,b3 with edges a1-b1, b1-a2, a2-b2, b2-a3, a3-b3, b3-a1
    
    # To avoid 6-cycles, we can use a "linear" structure.
    
    # Let's try: A = [0,1,2,3,4,5], B = [6,7,8,9,10,11]
    # Connect:
    # 0-6, 0-7, 0-8
    # 1-6, 1-7, 1-9
    # 2-7, 2-8, 2-9
    # 3-8, 3-9, 3-10
    # 4-9, 4-10, 4-11
    # 5-6, 5-10, 5-11
    
    # Check degrees: each A vertex has deg 3, each B vertex:
    # 6: 0,1,5 -> 3
    # 7: 0,1,2 -> 3
    # 8: 0,2,3 -> 3
    # 9: 1,2,3,4 -> 4
    # 10: 3,4,5 -> 3
    # 11: 4,5 -> 2  -- too low
    
    # Let me fix this. I need a 3-regular bipartite graph.
    
    # Use a known 3-regular bipartite graph on 12 vertices.
    # The (3,6)-cage is Heawood (14 vertices). For 12 vertices, 
    # let's try to construct one.
    
    # Actually, let me just try random 3-regular bipartite graphs and check cycle lengths.
    
    import random
    import sys
    
    def get_cycles(graph, n):
        """Find all simple cycles in a graph."""
        cycles = set()
        
        def dfs(start, current, path, visited):
            if time.time() > deadline:
                return
            if current == start and len(path) > 2:
                # Normalize cycle
                cycle = tuple(sorted(path))
                cycles.add(cycle)
                return
            for neighbor in graph[current]:
                if neighbor == start and len(path) > 2:
                    cycle = tuple(sorted(path))
                    cycles.add(cycle)
                    return
                if neighbor not in visited and neighbor > start:
                    # Only visit vertices > start to avoid duplicates
                    dfs(start, neighbor, path + [neighbor], visited | {neighbor})
        
        for start in range(n):
            dfs(start, start, [start], {start})
        
        return cycles
    
    def get_cycle_lengths(cycles):
        """Get distinct cycle lengths from a set of cycles."""
        lengths = set()
        for cycle in cycles:
            lengths.add(len(cycle))
        return lengths
    
    def is_valid(graph, n):
        """Check if graph is valid: min degree >= 3, triangle-free, has a cycle."""
        # Check degree
        for i in range(n):
            if len(graph[i]) < 3:
                return False
        # Check triangle-free
        for i in range(n):
            for j in graph[i]:
                for k in graph[j]:
                    if k == i:
                        continue
                    if k in graph[i]:
                        return False
        # Check has cycle (for connected component, check if edges >= vertices in some component)
        # Simpler: just check if there's any cycle
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                # BFS to check for cycle
                queue = deque([(i, -1)])
                visited[i] = True
                has_cycle = False
                while queue:
                    node, parent = queue.popleft()
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append((neighbor, node))
                        elif neighbor != parent:
                            has_cycle = True
                if has_cycle:
                    return True
        return False
    
    def count_distinct_cycle_lengths(graph, n):
        """Count distinct cycle lengths."""
        cycles = get_cycles(graph, n)
        if not cycles:
            return 0
        lengths = get_cycle_lengths(cycles)
        return len(lengths)
    
    # Try to find a 3-regular bipartite graph with minimal distinct cycle lengths
    best_edges = None
    best_score = float('inf')
    
    # First, try the two K_{3,3} construction as baseline
    edges = []
    for i in range(3):
        for j in range(3):
            edges.append([i, j+3])
            edges.append([i+6, j+9])
    
    # Build adjacency
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    adj = build_adj(edges, n)
    if is_valid(adj, n):
        score = count_distinct_cycle_lengths(adj, n)
        best_score = score
        best_edges = edges.copy()
    
    # Now try random 3-regular bipartite graphs
    A = list(range(6))
    B = list(range(6, 12))
    
    for trial in range(500):
        if time.time() > deadline - 5:
            break
        
        # Generate random 3-regular bipartite graph
        # Each vertex in A connects to 3 random vertices in B
        # Use configuration model with rejection
        
        valid = True
        adj_b = {v: [] for v in B}
        adj_a = {v: [] for v in A}
        
        # Shuffle B for each A vertex
        for a in A:
            neighbors = random.sample(B, 3)
            adj_a[a] = neighbors
            for b in neighbors:
                adj_b[b].append(a)
        
        # Check if each B vertex has degree exactly 3
        for b in B:
            if len(adj_b[b]) != 3:
                valid = False
                break
        
        if not valid:
            continue
        
        # Build edge list
        edges = []
        for a in A:
            for b in adj_a[a]:
                edges.append([a, b])
        
        # Build adjacency for checking
        adj = build_adj(edges, n)
        
        if not is_valid(adj, n):
            continue
        
        # Count distinct cycle lengths
        score = count_distinct_cycle_lengths(adj, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges.copy()
            
            if best_score == 1:
                break
    
    # Return best found
    if best_edges is None:
        # Fallback to two K_{3,3}
        best_edges = []
        for i in range(3):
            for j in range(3):
                best_edges.append([i, j+3])
                best_edges.append([i+6, j+9])
    
    # Convert to proper format
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return sorted(result)
