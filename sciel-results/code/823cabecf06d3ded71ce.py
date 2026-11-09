import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Moore bound for k=4, g=10 is approx 485 for general, 
    # but bipartite lower bound is often cited differently.
    # Prompt says lower bound 243. 
    # Let's start from 244.
    
    n_min = 244
    # n must be even for bipartite regular graph
    if n_min % 2 != 0:
        n_min += 1
        
    # We will try to find a graph with n vertices.
    # If we can't find one quickly, we increase n.
    
    # To be safe and efficient, we can pre-check small n.
    # But random search might be slow for small n if probability is low.
    # Let's start from a reasonable n where random graphs are likely to have high girth.
    # For k=4, girth 10, the probability of a random 4-reg graph having girth < 10 decays as n grows.
    # n=320 is known to exist. 
    # Let's try n from 244 up to maybe 600.
    
    # Optimization: If we find a valid graph, return it.
    # We should prioritize finding *any* valid graph with small n.
    
    # Heuristic: The number of 4-regular graphs on n vertices is huge.
    # The probability that a random 4-regular graph has girth >= 10 is non-negligible for n > 300.
    
    # Let's iterate n from 244 to 1000.
    # For each n, try a few random instances.
    
    def generate_random_bipartite_regular(n, k):
        """
        Generates a random k-regular bipartite graph with n vertices (n/2 in each part).
        Returns adjacency list or None if fails (e.g., multiple edges).
        """
        if n % 2 != 0:
            return None
        half = n // 2
        # Stubs for left side (0 to half-1)
        # Stubs for right side (half to n-1)
        
        # Configuration model:
        # Left stubs: list of (vertex, stub_id)
        # Right stubs: list of (vertex, stub_id)
        
        left_vertices = list(range(half))
        right_vertices = list(range(half, n))
        
        # Each vertex has k stubs
        left_stubs = []
        for v in left_vertices:
            for _ in range(k):
                left_stubs.append(v)
                
        right_stubs = []
        for v in right_vertices:
            for _ in range(k):
                right_stubs.append(v)
        
        # Shuffle right stubs
        random.shuffle(right_stubs)
        
        # Pair them
        edges = []
        # To avoid multiple edges, we need to check.
        # If we detect multiple edges, we can try to fix or restart.
        # For simplicity, restart if multiple edges found.
        
        edge_set = set()
        for i in range(len(left_stubs)):
            u = left_stubs[i]
            v = right_stubs[i]
            if (u, v) in edge_set:
                return None # Restart
            edge_set.add((u, v))
            edges.append((u, v))
            
        return edges

    def check_girth(edges, n, g):
        """
        Checks if the graph has girth at least g.
        Uses non-backtracking walk expansion.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length < g
        # Max depth for BFS is floor((g-1)/2)
        max_depth = (g - 1) // 2
        
        # For each vertex, do BFS up to max_depth
        # If we find a node that is already visited in the current path, we have a cycle.
        # Actually, standard girth check:
        # A graph has girth >= g iff for every vertex v, the number of non-backtracking walks
        # of length l is k*(k-1)^(l-1) for l <= floor((g-1)/2).
        
        for v in range(n):
            # State: (current_node, prev_node)
            # We want to see if any path returns to v or hits a previous node in the path.
            # Since we are checking for ANY short cycle, we can just check if the tree expands perfectly.
            # If it expands perfectly up to depth max_depth, there are no cycles of length <= 2*max_depth + 1.
            # Since 2*max_depth + 1 >= g (actually equal if g odd, or g-1 if g even?), 
            # Let's check: g=10, max_depth=4. 2*4+1=9. 
            # If no cycle of length <= 9, then girth >= 10.
            
            visited = set()
            # Queue items: (node, parent, depth)
            queue = [(v, -1, 0)]
            visited.add((v, -1))
            
            while queue:
                node, parent, depth = queue.pop(0)
                if depth == max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    # If we hit the start vertex v at depth > 0, we have a cycle of length depth+1
                    if neighbor == v and depth + 1 < g:
                        return False
                    # If we hit a node that is already in the visited set for this BFS tree?
                    # No, a non-backtracking walk is a path. 
                    # If we reach a node (node, parent) that we've seen before, it's a cycle.
                    # But since we are expanding a tree, we just need to ensure we don't hit v early.
                    # And we don't hit any node in the current path.
                    # However, if the tree is simple (no collisions), it implies no short cycles.
                    
                    # Actually, simpler check:
                    # If the number of nodes at depth d is exactly k*(k-1)^(d-1) for d=1..max_depth,
                    # and no node appears twice in the whole tree (which would imply a cycle shorter than 2*max_depth+1),
                    # then girth > 2*max_depth.
                    
                    # Let's just store all visited (node, parent) pairs.
                    # If (neighbor, node) is in visited, we have a cycle.
                    if (neighbor, node) in visited:
                        # Cycle found. Length is depth + 1 + (depth of neighbor in tree? No)
                        # Actually, if (neighbor, node) is in visited, it means we reached 'neighbor' coming from 'node' before?
                        # No, visited stores (current, parent).
                        # If we are at 'node' coming from 'parent', and we go to 'neighbor',
                        # and 'neighbor' was previously visited coming from 'node', that's backtracking (handled).
                        # If 'neighbor' was visited coming from 'some_other', then we have a cycle.
                        # The length of the cycle is depth + 1 + depth_of(neighbor) ?
                        # This is getting complicated.
                        
                        # Easier: Just check if 'neighbor' is in the set of all nodes visited in this BFS.
                        # If 'neighbor' is in visited_nodes (regardless of parent), and neighbor != parent,
                        # then we have a cycle.
                        # But we must be careful: if neighbor is an ancestor, it's a cycle.
                        # If neighbor is a node in a different branch, it's a cycle.
                        
                        # Let's keep a set of all nodes visited so far in this BFS.
                        # If neighbor is in that set, and neighbor != parent, we have a cycle.
                        # The length of the cycle is (depth + 1) + (depth of neighbor) ? No.
                        # If we just want to know IF girth < g, we can stop as soon as we find ANY cycle of length < g.
                        # If neighbor is in visited_nodes, the cycle length is at most depth + 1 + depth_neighbor.
                        # Since depth <= max_depth and depth_neighbor <= max_depth,
                        # cycle length <= 2*max_depth + 1 = g-1 (for g=10).
                        # So if we find a collision, girth < g.
                        
                        return False
                    
                    visited.add((neighbor, node))
                    # Also track node itself for collision detection
                    # We need a set of all nodes visited.
                    # Let's use a separate set for nodes.
                    if neighbor in visited_nodes:
                        return False
                    visited_nodes.add(neighbor)
                    
                    queue.append((neighbor, node, depth + 1))
                    
            # Note: visited_nodes needs to be defined in the loop
            # I messed up the variable scope in the sketch.
        
        return True

    # Corrected check_girth
    def check_girth_correct(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = (g - 1) // 2
        
        for v in range(n):
            visited_nodes = set([v])
            # (node, parent)
            queue = [(v, -1, 0)]
            
            while queue:
                node, parent, depth = queue.pop(0)
                if depth == max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited_nodes:
                        # Found a cycle. 
                        # Since we are doing BFS, the cycle length is at most depth + 1 + (depth of neighbor).
                        # Actually, if neighbor is in visited_nodes, there is a path from v to neighbor.
                        # The new path is v->...->node->neighbor.
                        # The cycle is formed by the two paths from v to neighbor.
                        # Length = depth(node) + 1 + depth(neighbor).
                        # Both depths <= max_depth.
                        # So cycle length <= 2*max_depth + 1 < g.
                        return False
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, node, depth + 1))
        
        return True

    # Main Loop
    best_n = None
    best_edges = None
    
    # Try small n first
    # Probability of finding a valid graph is low for small n.
    # We should limit the number of attempts per n to avoid spending too much time on impossible n.
    
    for n in range(n_min, 1000, 2):
        if time.time() > deadline:
            break
            
        # Try up to 100 random graphs for this n
        attempts = 0
        max_attempts = 100 if n < 300 else 500
        
        while attempts < max_attempts and time.time() < deadline:
            edges = generate_random_bipartite_regular(n, k)
            if edges is not None:
                if check_girth_correct(edges, n, g):
                    best_n = n
                    best_edges = edges
                    break
            attempts += 1
            
        if best_n is not None:
            break
        # If we exhausted attempts for this n, increase n
        
    if best_n is not None:
        return best_n, best_edges
        
    # Fallback: Return the seed result if we didn't find anything better
    # This ensures we don't crash
    return construct_seed(k, g)

def construct_seed(k, g):
    # Copy of the original seed code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# Re-define construct to use the logic above, but ensure seed is called correctly if needed.
# The seed function is defined below construct, so we need to move it up or use a forward reference.
# In Python, functions are resolved at call time, so it's fine.
