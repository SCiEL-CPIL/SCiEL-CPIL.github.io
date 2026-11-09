"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _has_c6(edges, n):
    """Check if the graph contains a 6-cycle."""
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # For each pair of non-adjacent vertices (u, v), check if there's a path of length 4
    # between them that forms a 6-cycle with the edge... wait, no.
    # A 6-cycle is u1-u2-u3-u4-u5-u6-u1.
    # We can check for each pair of vertices (u, v) at distance 2, 
    # or more efficiently: for each edge (u, v), check if there is a path of length 4 
    # between u and v that doesn't use the edge (u, v) and doesn't repeat vertices.
    
    # Actually, a simpler approach: for each pair of vertices (a, b) that are NOT adjacent,
    # check if there exists a path a - x - y - z - w - b where all intermediate vertices are distinct
    # and not equal to a or b, and no three consecutive vertices form a triangle (to ensure simple cycle).
    # Actually, for a 6-cycle, we just need a path of length 5 between two non-adjacent vertices 
    # such that the path is simple (no repeated vertices) and the endpoints are not adjacent.
    # But wait, if a and b are not adjacent, and there's a simple path of length 5, that forms a 6-cycle.
    
    # Alternative: for each pair of non-adjacent vertices (a, b), check if there's a simple path of length 5.
    # This is expensive. Let's use a different approach.
    
    # For each vertex v, look at its neighbors. For each pair of neighbors (x, y) of v, 
    # if x and y are not adjacent, check if there's a path of length 3 from x to y 
    # that doesn't go through v.
    
    # Actually, the most efficient way for small graphs:
    # For each pair of non-adjacent vertices (a, b), do a BFS/DFS to find a path of exactly length 5.
    
    # Let's use a different strategy: enumerate all 6-cycles by checking for each 
    # ordered pair of non-adjacent vertices (a, b) if there's a simple path of length 5.
    
    # But this is O(n^2 * something). For n=40, n^2 = 1600, which is fine.
    
    # For each non-adjacent pair (a, b), check if there's a simple path of length 5.
    # We can do this with a DFS that tracks visited vertices and depth.
    
    def dfs_path_length_5(start, end, current, visited, depth):
        if depth == 5:
            return current == end
        if depth > 5:
            return False
        for neighbor in adj[current]:
            if neighbor not in visited:
                if neighbor == end and depth == 4:
                    # We can only end at end at depth 5, not depth 4
                    # Wait, path length 5 means 5 edges, so 6 vertices including start and end.
                    # Start is at depth 0, end should be at depth 5.
                    pass
                visited.add(neighbor)
                if dfs_path_length_5(start, end, neighbor, visited, depth + 1):
                    visited.remove(neighbor)
                    return True
                visited.remove(neighbor)
        return False
    
    # Actually, let me redo this. A path of length 5 from a to b means:
    # a - v1 - v2 - v3 - v4 - b
    # So we need to find v1, v2, v3, v4 such that:
    # a-v1, v1-v2, v2-v3, v3-v4, v4-b are all edges
    # and a, v1, v2, v3, v4, b are all distinct
    # and a and b are not adjacent
    
    for a in range(n):
        for b in range(a + 1, n):
            if b in adj[a]:
                continue  # a and b are adjacent, so a 6-cycle through them would need a path of length 5, but they're already connected
            # Check if there's a simple path of length 5 from a to b
            # v1 must be a neighbor of a
            # v4 must be a neighbor of b
            # v1, v2, v3, v4 must be distinct and not equal to a or b
            found = False
            for v1 in adj[a]:
                if v1 == b:
                    continue
                for v4 in adj[b]:
                    if v4 == a or v4 == v1:
                        continue
                    # Now we need a path v1 - v2 - v3 - v4 of length 3
                    # v2 is a neighbor of v1, v3 is a neighbor of v4
                    # v2 != v3, v2 != v4, v3 != v1, v2 != a, v3 != b, v2 != b, v3 != a
                    for v2 in adj[v1]:
                        if v2 == a or v2 == b or v2 == v4:
                            continue
                        for v3 in adj[v4]:
                            if v3 == a or v3 == b or v3 == v1 or v3 == v2:
                                continue
                            if v2 in adj[v3]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                return True
    return False

def _creates_c6(edge_set, u, v, n):
    """Check if adding edge (u, v) creates a 6-cycle."""
    # A 6-cycle containing edge (u, v) would be u - v - x - y - z - w - u
    # where x, y, z, w are distinct vertices not equal to u or v
    # and u-w, w-z, z-y, y-x, x-v are edges
    # So we need to find x, y, z, w such that:
    # x is adjacent to v
    # w is adjacent to u
    # x-y, y-z, z-w are edges
    # x, y, z, w are distinct and not u or v
    
    adj = [set() for _ in range(n)]
    for a, b in edge_set:
        adj[a].add(b)
        adj[b].add(a)
    
    for x in adj[v]:
        if x == u:
            continue
        for w in adj[u]:
            if w == v or w == x:
                continue
            # Need a path x - y - z - w of length 3
            for y in adj[x]:
                if y == u or y == v or y == w:
                    continue
                for z in adj[w]:
                    if z == u or z == v or z == x or z == y:
                        continue
                    if y in adj[z]:
                        return True
    return False

def construct():
    n = 40
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(42)
    
    # Start with a random dense graph and remove edges that create C6
    # Or start with no edges and add them greedily
    
    # Strategy: Start with a random ordering of all possible edges, 
    # and add an edge if it doesn't create a C6.
    # Then try to improve by removing one edge and adding two others.
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    edge_set = set()
    edge_list = []
    
    # Greedy: add edges in random order if they don't create a C6
    for u, v in all_edges:
        if time.time() > deadline:
            break
        # Check if adding (u, v) creates a C6
        # We need to check if there's a path of length 4 between u and v that doesn't use (u, v)
        # and forms a simple cycle with (u, v)
        
        # Quick check: if u and v already have a common neighbor, that's fine (triangle, not C6)
        # We need a path of length 4: u - a - b - c - v where a, b, c are distinct and not u or v
        
        creates = False
        # Build adjacency for current edge_set
        # For efficiency, maintain adjacency sets
        if not hasattr(construct, '_adj'):
            construct._adj = [set() for _ in range(n)]
        
        adj = construct._adj
        
        # Check if adding (u, v) creates a C6
        # We need to find a, b, c such that:
        # u-a, a-b, b-c, c-v are edges
        # a, b, c are distinct and not u or v
        
        # For each neighbor a of u:
        #   For each neighbor c of v:
        #     If a != c:
        #       Check if there's a path a - b - c of length 2 (i.e., common neighbor of a and c)
        #       where b != u, b != v, b != a, b != c
        
        for a in adj[u]:
            if a == v:
                continue
            for c in adj[v]:
                if c == u or c == a:
                    continue
                # Check if a and c have a common neighbor b
                # b != u, b != v, b != a, b != c
                common = adj[a] & adj[c]
                for b in common:
                    if b != u and b != v and b != a and b != c:
                        creates = True
                        break
                if creates:
                    break
            if creates:
                break
        
        if not creates:
            edge_set.add((u, v))
            edge_list.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
    
    # Now try to improve: remove an edge and add two others
    # This is complex, so let's try a simpler local search
    # Try to add edges that were skipped
    
    skipped = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
    random.shuffle(skipped)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for u, v in skipped:
            if time.time() > deadline - 5:
                break
            # Check if adding (u, v) creates a C6
            creates = False
            adj = construct._adj
            for a in adj[u]:
                if a == v:
                    continue
                for c in adj[v]:
                    if c == u or c == a:
                        continue
                    common = adj[a] & adj[c]
                    for b in common:
                        if b != u and b != v and b != a and b != c:
                            creates = True
                            break
                    if creates:
                        break
                if creates:
                    break
            if not creates:
                edge_set.add((u, v))
                edge_list.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                skipped.remove((u, v))
                improved = True
    
    # Verify no C6
    if _has_c6(edge_list, n):
        # Fallback to the simple construction
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges
    
    return edge_list

# EVOLVE-BLOCK-END
