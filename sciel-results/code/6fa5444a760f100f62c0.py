import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    # Initial candidate: 5 blocks of sizes 5, 5, 5, 5, 4
    # Blocks: [0-4], [5-9], [10-14], [15-19], [20-23]
    # Connect all pairs between adjacent blocks.
    blocks = [
        list(range(0, 5)),
        list(range(5, 10)),
        list(range(10, 15)),
        list(range(15, 20)),
        list(range(20, 24))
    ]
    
    candidate_edges = set()
    for i in range(len(blocks) - 1):
        for u in blocks[i]:
            for v in blocks[i+1]:
                if u % 2 != v % 2: # Ensure bipartite (Even-Odd)
                    if (u, v) not in base_edges and (v, u) not in base_edges:
                        candidate_edges.add((u, v))
                else:
                    # If same parity, we can't add it if we want to stay bipartite.
                    # But wait, the block boundaries might have same parity vertices.
                    # e.g., 4 (Even) and 5 (Odd).
                    # 0(E) to 5(O) is OK.
                    # 4(E) to 9(O) is OK.
                    # All pairs in adjacent blocks of a path are Even-Odd?
                    # Block 0: 0,1,2,3,4. Block 1: 5,6,7,8,9.
                    # 0(E) to 5(O). 0(E) to 6(E) -> Same parity.
                    # So we MUST filter.
                    pass
                    
    # Re-calculate candidate edges correctly
    candidate_edges = set()
    for i in range(len(blocks) - 1):
        for u in blocks[i]:
            for v in blocks[i+1]:
                if u % 2 != v % 2:
                    edge = (min(u,v), max(u,v))
                    if edge not in base_edges:
                        candidate_edges.add(edge)

    # Function to check diameter
    def get_diameter(edges_set):
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # BFS from each node
        max_d = 0
        for start in range(n):
            dist = {i: -1 for i in range(n)}
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            # Check if all reachable
            if -1 in dist.values():
                return float('inf')
            max_d = max(max_d, max(dist.values()))
        return max_d

    # Verify initial candidate
    current_edges = base_edges.union(candidate_edges)
    if get_diameter(current_edges) > 4:
        # Fallback to seed if construction fails (should not happen)
        return [[u, v] for u in range(n) for v in range(u+1, n) if (u-v)%2 and (u,v) not in base_edges]

    # Try to remove edges
    # Sort edges to have a deterministic order
    edges_list = list(candidate_edges)
    
    # We can try removing edges that are "redundant".
    # Iterate through the list.
    for i in range(len(edges_list)):
        e = edges_list[i]
        if e not in current_edges:
            continue
        # Try removing
        current_edges.discard(e)
        if get_diameter(current_edges) <= 4:
            # Keep removed
            continue
        else:
            # Restore
            current_edges.add(e)

    # Convert to list of lists
    result = [[u, v] for u, v in current_edges if (u, v) not in base_edges]
    
    # Ensure types are correct
    return [[int(u), int(v)] for u, v in result]

# EVOLVE-BLOCK-END
