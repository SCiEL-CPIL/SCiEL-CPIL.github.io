"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = P['n']
    # Build initial adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    best_branches = []
    best_paths = []
    
    # Helper: Find shortest path between u and v using only allowed nodes
    # allowed_nodes is a set of nodes that can be used as interiors (and endpoints if they are u/v)
    def find_shortest_path(u, v, allowed_nodes, adj):
        if u == v:
            return [u]
        if v not in allowed_nodes:
            return None
        # BFS
        visited = {u}
        queue = deque([(u, [u])])
        while queue:
            node, path = queue.popleft()
            if node == v:
                return path
            for nei in adj[node]:
                if nei not in visited:
                    # Interior nodes must be in allowed_nodes
                    # Endpoints u and v are already handled by the check above
                    # If nei is v, it's allowed. If nei is not u/v, it must be in allowed_nodes.
                    if nei == v or nei in allowed_nodes:
                        visited.add(nei)
                        queue.append((nei, path + [nei]))
        return None

    # Helper: Check if a set of branches is feasible and find the paths
    # This function tries to find disjoint paths for a given set of branches.
    # It uses a greedy approach: for each pair, find the shortest path in the current residual graph.
    # If a path is found, remove its interior nodes from the available set.
    # To maximize success, we can try different orderings or use backtracking if simple greedy fails.
    # Given the small size (n=24), we can use a simple backtracking for path assignment if greedy fails,
    # but let's first try a robust greedy with randomization/restarts.
    
    def try_construct(branches):
        """
        Given a list of branches, try to find disjoint paths between all pairs.
        Returns (True, paths_dict) if successful, (False, None) otherwise.
        """
        k = len(branches)
        if k < 2:
            return False, None
            
        # Initialize available nodes: all nodes except branches
        available = set(range(n)) - set(branches)
        
        # We need to assign a path to each pair (i, j) in combinations(branches, 2)
        # To make this easier, we can sort pairs by "difficulty" (e.g., number of common neighbors)
        # or just use a fixed order. Randomized order might help.
        
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((branches[i], branches[j]))
        
        # Shuffle pairs to try different orders? Or keep fixed?
        # Fixed order is deterministic. Random order requires multiple tries.
        # Let's try a few random orders of pairs.
        
        best_result = None
        
        for _ in range(100): # Try 100 random orderings
            current_available = set(available)
            paths_found = {}
            success = True
            shuffled_pairs = pairs[:]
            random.shuffle(shuffled_pairs)
            
            for u, v in shuffled_pairs:
                path = find_shortest_path(u, v, current_available, adj)
                if path is None:
                    success = False
                    break
                # Remove interior nodes from current_available
                for node in path[1:-1]:
                    if node in current_available:
                        current_available.remove(node)
                paths_found[(u, v)] = path
                
            if success:
                # Check if this is better (it's the same size, but maybe we want to store one)
                # Since we just need one valid solution, we can return immediately
                return True, paths_found
                
        return False, None

    # Strategy: Greedy addition of branches
    # Start with a small set, try to add vertices one by one.
    # To get a large set, we should start with high-degree vertices.
    
    # Sort vertices by degree descending
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    current_branches = []
    
    # Try to build a large set
    # We will iterate through candidates and try to add them.
    # If adding a candidate fails, we skip it.
    # This is a heuristic. To improve, we can restart with different initial seeds or randomizations.
    
    # Let's do multiple restarts with different random seeds for the path-finding order
    # and maybe different initial branch sets.
    
    max_branches_found = 0
    best_solution = None
    
    # Restart loop
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize branches with top few vertices, maybe randomize the order of addition
        # or pick a random subset of top vertices to start?
        # Let's try starting with the top 4 vertices, but shuffle the candidate list slightly.
        
        # Candidate list: all vertices
        candidates = sorted_vertices[:]
        # Randomize the order of candidates we try to add, but bias towards high degree?
        # Actually, the greedy addition order matters.
        # Let's try adding vertices in the order of sorted_vertices, but with some random swaps.
        
        # Simpler: Just try to add vertices in sorted order.
        # To get diversity, we can randomly skip some high-degree vertices in early rounds?
        # Or just rely on the randomization in try_construct.
        
        # Let's try a different strategy:
        # Pick a random starting clique of size 2 or 3?
        # No, let's stick to the greedy addition but with randomization in the "try to add" phase.
        
        current_branches = []
        
        # Try to add vertices one by one
        # We'll iterate through all vertices in sorted order
        for v in sorted_vertices:
            if v in current_branches:
                continue
                
            # Try to add v
            new_branches = current_branches + [v]
            
            # Check feasibility
            feasible, paths = try_construct(new_branches)
            
            if feasible:
                current_branches = new_branches
                # Update best if larger
                if len(current_branches) > max_branches_found:
                    max_branches_found = len(current_branches)
                    best_solution = (current_branches[:], paths)
                
                # If we have time, we can try to add more, but the loop continues.
                # We don't break because we want to see if we can add even more.
            else:
                # Skip v
                pass
        
        # End of restart
        
    # If no solution found with > 2 branches, fallback to simple edge
    if best_solution is None:
        u, v = P['edges'][0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    else:
        branches, paths_dict = best_solution
        # Convert paths_dict to list of lists for the required format
        # The key is (u, v) where u < v in the branches list?
        # The contract says "one simple path for each unordered pair of branches".
        # The keys in paths_dict are (u, v) where u and v are from the branches list.
        # We need to return a list of paths. The order doesn't matter as long as it covers all pairs.
        # Let's just extract the values.
        paths_list = list(paths_dict.values())
        
        # Ensure types are Python int and list
        branches_list = [int(b) for b in branches]
        paths_final = [[int(x) for x in path] for path in paths_list]
        
        return {'branches': branches_list, 'paths': paths_final}

# EVOLVE-BLOCK-END
