import time
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s safety margin out of 90s
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute degrees for initial greedy
    degrees = [len(adj[i]) for i in range(n)]
    
    best_path = []
    best_len = 0
    
    def greedy_path(start_node, visited_set=None):
        if visited_set is None:
            visited_set = set()
        path = [start_node]
        visited_set.add(start_node)
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited_set]
            if not neighbors:
                break
            # Choose neighbor with lowest degree (Warnsdorff's rule)
            # Tie-break by node id for determinism
            next_node = min(neighbors, key=lambda x: (degrees[x], x))
            path.append(next_node)
            visited_set.add(next_node)
        return path

    # Initial pass: Try greedy from every node
    for i in range(n):
        if time.time() > deadline:
            break
        path = greedy_path(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            
    # Local Search / Randomized Extension Strategy
    # We will try to extend the current best path or find better paths by random restarts with backtracking
    # Since the graph is small (100 nodes), we can afford some backtracking.
    
    rng = random.Random(748)
    
    def backtrack_path(current_node, visited, path, limit=20):
        # Returns a path extended from current_node, or None if stuck
        # This is a simplified DFS that stops early if too deep to avoid explosion, 
        # but we use it to try to extend a partial path.
        
        # Actually, for a simple path in a sparse graph, a simple DFS might work if we are smart.
        # Let's implement a proper backtracking search that tries to find a path of length > best_len
        # starting from a specific node, but limited in depth/time.
        pass

    # A more effective strategy for "Long paths in sparse graphs":
    # 1. Get a good initial path (done).
    # 2. Try to extend the ends of the path.
    # 3. Try to "slide" the path or find a different path by random restarts with a different heuristic.
    
    # Let's implement a randomized restart strategy with a different heuristic:
    # Random neighbor selection, but with a bias towards unvisited nodes with higher connectivity potential?
    # Or simply: Try to find a path that covers more nodes by restarting with random start nodes 
    # and using a "lookahead" heuristic.
    
    # Heuristic: Minimize the number of unvisited neighbors of the next node (Warnsdorff) is good.
    # Let's try to improve by checking if we can extend the ends of the best_path.
    
    def try_extend_ends(path):
        # Try to extend the start or end of the path
        # We need to be careful: the path is fixed in the middle.
        # We can only extend if there is an edge from path[0] to an unvisited node, 
        # or from path[-1] to an unvisited node.
        # But wait, the path is a simple path. The "unvisited" nodes are those not in the path.
        # If we extend, we just add to the list.
        
        # However, the greedy already did this. The greedy stops when there are NO unvisited neighbors.
        # So the ends of the greedy path are "dead ends" in the subgraph induced by unvisited nodes + path ends?
        # No, the greedy stops when adj[last] - seen is empty.
        # So we cannot simply extend the ends of a greedy path unless we change the internal structure.
        
        # Therefore, we need a different search.
        
        # Strategy: Randomized Backtracking Search
        # Start from a random node.
        # Use a stack.
        # Choose next node randomly from unvisited neighbors.
        # If stuck, backtrack.
        # Limit the depth of backtracking to keep it fast.
        
        pass

    # Let's implement a randomized DFS with backtracking limit.
    # This is more likely to find longer paths than pure greedy if the greedy gets trapped.
    
    def randomized_dfs(start_node, target_len):
        # We want to find a path of length >= target_len
        # If we find one, we return it.
        
        # State: path, visited
        # We use a stack of states.
        # To avoid memory explosion, we limit the number of backtracks.
        
        # Actually, with n=100, a full DFS might be too slow if the graph is connected and dense enough.
        # But p=0.018 is sparse. The largest component is likely small or medium.
        # Let's check the size of the largest component first to see if a full search is feasible.
        
        pass

    # Let's just do many randomized restarts of a "smart" greedy.
    # Heuristic: Choose neighbor that has the most unvisited neighbors (Max Degree among unvisited)
    # This is the opposite of Warnsdorff. Sometimes useful.
    
    def greedy_path_max_degree(start_node):
        visited = set([start_node])
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            # Choose neighbor with highest degree in the remaining graph
            # Degree in remaining graph = len(adj[x] - visited)
            next_node = max(neighbors, key=lambda x: (len(adj[x] - visited), x))
            path.append(next_node)
            visited.add(next_node)
        return path

    for i in range(n):
        if time.time() > deadline:
            break
        path = greedy_path_max_degree(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # One more heuristic: Random restarts with random choices, but with a "bias" 
    # to avoid dead ends.
    # If a node has only 1 unvisited neighbor, we MUST go there (forced move).
    # If a node has >1, we pick one.
    # This is basically the same as greedy but with random tie-breaking or random choice when >1.
    
    def greedy_path_random_forced(start_node):
        visited = set([start_node])
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break
            
            # Check for forced moves: neighbors that have only 1 unvisited neighbor (which is 'last')
            # Actually, if a neighbor x has len(adj[x] - visited) == 1, and that neighbor is 'last',
            # then if we don't go to x now, we can never go to x later (unless we come back, which we can't in a simple path).
            # Wait, if we don't go to x, x becomes isolated from the unvisited set?
            # No, x might have other neighbors that are unvisited.
            # len(adj[x] - visited) counts unvisited neighbors of x.
            # If len(adj[x] - visited) == 1, that one neighbor must be 'last' (since x is a neighbor of last).
            # So if we don't pick x, x will have 0 unvisited neighbors in the future (because 'last' will be visited).
            # So x would be stranded.
            # Therefore, if there is a neighbor x with len(adj[x] - visited) == 1, we MUST pick x.
            
            forced = [x for x in neighbors if len(adj[x] - visited) == 1]
            if forced:
                next_node = forced[0]
            else:
                # Pick randomly from neighbors
                next_node = rng.choice(neighbors)
                
            path.append(next_node)
            visited.add(next_node)
        return path

    for i in range(n):
        if time.time() > deadline:
            break
        path = greedy_path_random_forced(i)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Try to improve best_path by local search:
    # Pick a random vertex in the path, remove it, and try to reconnect?
    # This is complex.
    
    # Instead, let's do a few iterations of "2-opt" style or just more random restarts.
    # Given the time limit, more random restarts of the forced-move greedy is probably the best bang for the buck.
    
    # Let's run 1000 more random restarts of the forced-move greedy, starting from random nodes.
    # But we already did all nodes.
    # Let's vary the random seed for the non-forced choices.
    
    for _ in range(500):
        if time.time() > deadline:
            break
        start_node = rng.randint(0, n-1)
        path = greedy_path_random_forced(start_node)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    return best_path

# EVOLVE-BLOCK-END
