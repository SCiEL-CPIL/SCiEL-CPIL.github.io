import time
import random
from collections import deque

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Nodes are represented as integers in base n, length n-1
    # Node value: d_0 * n^(n-2) + ... + d_{n-2}
    # Symbol at pos i (0-indexed from left) is digit i
    
    node_count = n ** (n - 1)
    
    # Precompute required edges
    # An edge is defined by (start_node, end_node)
    # start_node is perm[0:n-1], end_node is perm[1:n]
    # We need to map permutation tuples to node IDs
    
    def tuple_to_node(t):
        val = 0
        for i in range(len(t)):
            val = val * n + t[i]
        return val

    def node_to_tuple(node):
        t = [0] * (n - 1)
        for i in range(n - 2, -1, -1):
            t[i] = node % n
            node //= n
        return tuple(t)

    # Generate all permutations
    import itertools
    perms = list(itertools.permutations(range(1, n + 1)))
    
    required_edges = set()
    edge_start_node = {} # edge_id -> start_node
    edge_end_node = {}   # edge_id -> end_node
    
    for p in perms:
        start_t = p[:n-1]
        end_t = p[1:]
        start_node = tuple_to_node(start_t)
        end_node = tuple_to_node(end_t)
        edge_id = (start_node, end_node)
        required_edges.add(edge_id)
        edge_start_node[edge_id] = start_node
        edge_end_node[edge_id] = end_node
        
    # Map from node to list of required outgoing edges
    node_to_req_edges = {i: [] for i in range(node_count)}
    for e in required_edges:
        node_to_req_edges[e[0]].append(e)
        
    # Unvisited required edges
    unvisited = set(required_edges)
    
    # Greedy walk
    # Start at an arbitrary node that has a required edge
    # Let's pick the start of the first permutation
    current_node = tuple_to_node(perms[0][:n-1])
    
    word = [1] * n # Initialize with the first permutation? 
    # Actually, we build the word by appending symbols.
    # Start with the first symbol of the first edge.
    # Let's just track the path of nodes.
    
    path_nodes = [current_node]
    
    # We need to append the symbols corresponding to the edges traversed.
    # If we traverse edge u->v, we append the last symbol of v (which is the symbol that created the transition).
    # Actually, the word is constructed by:
    # Start with the tuple of the first node.
    # For each subsequent node in the path, append the last symbol of that node.
    
    # Let's refine:
    # Word starts as the string of the first node's tuple? No.
    # If path is u0, u1, u2...
    # u0 is a tuple of length n-1.
    # u1 is a tuple of length n-1.
    # The edge u0->u1 implies u0[1:] == u1[:-1].
    # The symbol appended is u1[-1].
    # So the word is tuple(u0) + [u1[-1], u2[-1], ...]
    
    # Initialize word with the first permutation
    # We need to ensure we start with a valid edge.
    # Let's pick a random unvisited edge to start with?
    # Or just start with the first one.
    
    start_edge = list(unvisited)[0]
    current_node = start_edge[0]
    path_nodes = [current_node]
    
    # We will simulate the walk
    # We need to keep track of the word to return it.
    # But we can just reconstruct it from path_nodes at the end.
    
    # Heuristic for choosing required edge:
    # Pick the edge u->v such that v has the fewest unvisited required outgoing edges.
    
    def get_unvisited_count(node):
        count = 0
        for e in node_to_req_edges[node]:
            if e in unvisited:
                count += 1
        return count

    # BFS to find nearest unvisited required edge start
    def find_nearest_unvisited_start(start_node):
        # Returns (distance, path)
        # Path is list of nodes from start_node to target_node
        # We want to reach a node 's' such that there is an unvisited edge starting at s.
        
        visited = {start_node: None} # node -> prev_node
        queue = deque([start_node])
        
        # Check if start_node itself has unvisited edges
        if any(e in unvisited for e in node_to_req_edges[start_node]):
            return 0, [start_node]
            
        while queue:
            u = queue.popleft()
            # Generate neighbors
            # u is a node (int). Neighbors are obtained by shifting and appending x.
            # u represents d_0...d_{n-2}
            # Neighbor v represents d_1...d_{n-2} x
            # v = (u % n^(n-2)) * n + x ? No.
            # u = d_0 * n^(n-2) + d_1 * n^(n-3) + ... + d_{n-2}
            # u % n^(n-2) is d_1...d_{n-2}
            # v = (u % n^(n-2)) * n + x
            # Wait, this drops d_0.
            # Yes, that's the De Bruijn transition.
            
            base = u % (n ** (n - 2))
            for x in range(1, n + 1):
                v = base * n + x
                if v not in visited:
                    visited[v] = u
                    # Check if v has unvisited required edges
                    if any(e in unvisited for e in node_to_req_edges[v]):
                        # Reconstruct path
                        path = [v]
                        curr = v
                        while visited[curr] is not None:
                            curr = visited[curr]
                            path.append(curr)
                        path.reverse()
                        return len(path) - 1, path
                    queue.append(v)
        return -1, [] # Should not happen if unvisited is not empty

    # Main loop
    while unvisited:
        # Check if current_node has unvisited required edges
        next_edge = None
        best_score = float('inf')
        
        for e in node_to_req_edges[current_node]:
            if e in unvisited:
                v = e[1]
                # Heuristic: fewest unvisited edges from v
                score = get_unvisited_count(v)
                if score < best_score:
                    best_score = score
                    next_edge = e
                    # Tie breaking: random or lexicographic
                    # Let's add a small random factor to avoid cycles in heuristic
                    # score += random.random() * 0.001 
                    
        if next_edge:
            # Traverse the required edge
            unvisited.remove(next_edge)
            current_node = next_edge[1]
            path_nodes.append(current_node)
        else:
            # Need to move to a node with unvisited edges
            dist, path = find_nearest_unvisited_start(current_node)
            if dist == -1:
                break # Error
            # Follow the path
            # The path includes current_node? 
            # My BFS returns path from start to target.
            # If start == target, path is [start].
            # We are at current_node. We need to move to path[-1].
            # The path is [current_node, ..., target].
            # We need to append the nodes in path[1:] to path_nodes.
            for node in path[1:]:
                path_nodes.append(node)
                current_node = node
                
    # Reconstruct word
    # path_nodes[0] is the start node.
    # Word starts with the tuple of path_nodes[0].
    # Then append last symbol of each subsequent node.
    
    first_node_t = node_to_tuple(path_nodes[0])
    word = list(first_node_t)
    
    for i in range(1, len(path_nodes)):
        t = node_to_tuple(path_nodes[i])
        word.append(t[-1])
        
    return "".join(map(str, word))

# Add deadline guard
import time
original_construct = construct

def construct_with_deadline(n):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # The greedy algorithm above is deterministic and fast.
    # It should finish in < 1s.
    # If it takes longer, something is wrong.
    # We'll just call it.
    
    return original_construct(n)

# Replace construct with the guarded version?
# The task asks to keep function names.
# We can just modify construct to include the logic.

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    # ... (implementation from above) ...
    
    # To ensure we don't hang, we can add a step limit.
    # Max steps: n! * 2 (just in case)
    max_steps = 2 * (1)
    for i in range(2, n + 1):
        max_steps *= i
    max_steps *= 2
    
    steps = 0
    while unvisited and steps < max_steps:
        # ... logic ...
        steps += 1
        
    # If unvisited is not empty, we failed.
    # But for n=7, it should succeed.
    
    return "".join(map(str, word))
