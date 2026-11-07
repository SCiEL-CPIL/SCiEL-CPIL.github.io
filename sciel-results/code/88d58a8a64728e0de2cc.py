"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for fast lookup
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    def is_valid_configuration(branches, paths_dict):
        """Validates the entire configuration."""
        if len(set(branches)) != len(branches):
            return False
        
        # Check all required pairs have paths
        for u, v in combinations(branches, 2):
            if (u, v) not in paths_dict and (v, u) not in paths_dict:
                return False
                
        used_interiors = set()
        branch_set = set(branches)
        for key, path in paths_dict.items():
            if len(path) < 2:
                return False
            u, v = key
            if path[0] != u and path[0] != v:
                return False
            if path[-1] != u and path[-1] != v:
                return False
            if path[0] == path[-1]:
                return False
                
            # Check adjacency
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
            
            # Check interiors are disjoint and avoid branches
            interiors = path[1:-1]
            for node in interiors:
                if node in branch_set:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(interiors)
            
        return True

    def find_disjoint_paths(branches, time_limit=0.5):
        """
        Attempts to find vertex-disjoint paths between all pairs in 'branches'.
        Returns a dict of paths or None if failed.
        Uses randomized greedy with backtracking/retries.
        """
        if len(branches) < 2:
            return {}
        
        # We need to find paths for all pairs.
        # Strategy: 
        # 1. Identify all pairs.
        # 2. Randomize the order of pairs.
        # 3. For each pair, find a path avoiding current forbidden nodes (other branches + used interiors).
        # 4. If stuck, retry with different random seeds/orders.
        
        pairs = list(combinations(branches, 2))
        best_paths = None
        attempts = 0
        max_attempts = 50 # Limit retries per branch set
        
        while time.time() < start_time + deadline and attempts < max_attempts:
            attempts += 1
            current_forbidden = set()
            # Pre-populate forbidden with all branch vertices? 
            # No, we handle endpoints in the path finder.
            # But interiors cannot touch ANY branch.
            # So for a path (u,v), forbidden = current_interiors + (branches - {u,v})
            
            found_paths = {}
            success = True
            shuffled_pairs = pairs[:]
            random.shuffle(shuffled_pairs)
            
            # Optimization: Sort pairs by "difficulty" (fewest available neighbors?) 
            # For now, random is fine with retries.
            
            for u, v in shuffled_pairs:
                # Define forbidden set for this specific path
                path_forbidden = set(current_forbidden)
                for b in branches:
                    if b != u and b != v:
                        path_forbidden.add(b)
                
                # Find a simple path from u to v avoiding path_forbidden
                # Use BFS with randomization to find different paths
                path = find_path_randomized(u, v, path_forbidden, adj, max_iter=100)
                
                if path is None:
                    success = False
                    break
                
                found_paths[(u, v)] = path
                # Update forbidden with interiors
                current_forbidden.update(path[1:-1])
            
            if success:
                # Verify full validity (just in case)
                if is_valid_configuration(branches, found_paths):
                    return found_paths
            # If failed, the loop restarts with a new random shuffle
            
        return None

    def find_path_randomized(u, v, forbidden, adj, max_iter=100):
        """
        Finds a simple path from u to v avoiding 'forbidden'.
        Uses randomized BFS/DFS to escape local traps.
        """
        if u in forbidden or v in forbidden:
            return None
        if u == v:
            return [u]
        
        # Try simple BFS first (fastest)
        # Then try randomized walks if BFS fails? 
        # Actually, BFS finds *a* path. If BFS fails, no path exists in the static graph 
        # *given the forbidden set*. 
        # Wait, if BFS fails, it means u and v are disconnected in G \ forbidden.
        # In that case, no path exists. We don't need randomized retries for a single pair 
        # if the forbidden set is fixed. 
        # The randomization needs to be on the *order* in which we assign paths, 
        # because the choice of path for pair 1 affects the forbidden set for pair 2.
        # So, for a FIXED forbidden set, BFS is optimal (finds shortest, most direct).
        # Shortest paths are often better for disjointness (less "wasted" vertices).
        
        # So, find_path_randomized is just BFS here. 
        # The "randomized" part is in the outer loop shuffling the pairs.
        
        if u == v:
            return [u]
            
        visited = {u}
        parent = {u: None}
        queue = deque([u])
        
        while queue:
            node = queue.popleft()
            if node == v:
                # Reconstruct path
                path = []
                curr = v
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path
            
            for neighbor in adj[node]:
                if neighbor in forbidden:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None

    def find_max_clique_seed(adj, n):
        """
        Finds a large clique in the graph to use as a seed.
        Uses a simple greedy heuristic with random restarts.
        """
        best_clique = []
        for _ in range(50): # 50 random restarts
            # Randomly order vertices
            vertices = list(range(n))
            random.shuffle(vertices)
            clique = []
            for v in vertices:
                # Check if v is connected to all in clique
                if all(v in adj[u] for u in clique):
                    clique.append(v)
            if len(clique) > len(best_clique):
                best_clique = clique
        return best_clique

    # --- Main Logic ---
    
    # 1. Get a good seed from a large clique
    seed_clique = find_max_clique_seed(adj, n)
    
    # If clique is too small, pad with random neighbors or just use it.
    # We need at least 2 branches.
    if len(seed_clique) < 2:
        # Fallback to any edge
        if edges:
            seed_clique = [edges[0][0], edges[0][1]]
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}

    best_branches = seed_clique[:]
    best_paths = None
    
    # Try to solve for the seed clique
    paths = find_disjoint_paths(best_branches)
    if paths:
        best_paths = paths
    else:
        # If the clique itself is infeasible (rare for small cliques, but possible),
        # try shrinking it or picking a different subset.
        # For now, if the best clique fails, try smaller subsets of it.
        for size in range(len(seed_clique) - 1, 1, -1):
            subset = seed_clique[:size]
            paths = find_disjoint_paths(subset)
            if paths:
                best_branches = subset
                best_paths = paths
                break
        if not best_paths:
            # Absolute fallback: 2 nodes
            best_branches = [edges[0][0], edges[0][1]]
            best_paths = {(best_branches[0], best_branches[1]): [best_branches[0], best_branches[1]]}

    current_branches = best_branches[:]
    current_paths = best_paths
    best_size = len(current_branches)
    
    # 2. Local Search / Hill Climbing
    # We will try to add new branches, or swap branches.
    # To allow "different" structures, we also do random restarts of the local search.
    
    search_deadline = start_time + deadline - 5.0 # Leave 5s for final validation
    
    while time.time() < search_deadline:
        improved = False
        
        # Generate candidates for addition
        candidates = [i for i in range(n) if i not in current_branches]
        random.shuffle(candidates)
        
        # Try to add one candidate
        added = False
        for new_b in candidates:
            new_branches = current_branches + [new_b]
            # Try to find paths for the new set
            # We can reuse existing paths if they don't conflict? 
            # No, the disjointness constraint is global. 
            # However, if we just add a branch, the old paths might still be valid 
            # IF the new paths don't touch the old interiors.
            # But find_disjoint_paths restarts from scratch. 
            # Optimization: We could try to keep old paths and only find new ones.
            # But for simplicity and robustness, we re-solve. 
            # To speed up, we can pass the old paths as a "hint" or just rely on the 
            # fact that BFS is fast.
            
            paths = find_disjoint_paths(new_branches, time_limit=0.2)
            if paths:
                current_branches = new_branches
                current_paths = paths
                best_size = len(current_branches)
                improved = True
                added = True
                break # Restart candidate loop with new state
        
        if not added:
            # Try swapping
            # Pick a random branch to remove and a random non-branch to add
            if len(current_branches) > 2:
                # Maybe try to swap a "bad" branch? 
                # For now, random swap.
                if random.random() < 0.5: # 50% chance to try swap if stuck
                    b_to_remove = random.choice(current_branches)
                    candidates = [i for i in range(n) if i not in current_branches]
                    if candidates:
                        b_to_add = random.choice(candidates)
                        new_branches = [b for b in current_branches if b != b_to_remove] + [b_to_add]
                        paths = find_disjoint_paths(new_branches, time_limit=0.2)
                        if paths:
                            # Only accept if size is same or larger? 
                            # Size is same. But maybe this opens up for future additions?
                            # Or maybe the current set is a dead end.
                            # Accept swap to escape local minima.
                            current_branches = new_branches
                            current_paths = paths
                            # best_size unchanged, but state changed.
                            improved = True # Mark as activity to prevent immediate restart
        
        if not improved:
            # Random Restart
            # Pick a new random seed (clique or random set)
            restart_seed_size = max(2, best_size - 2)
            # Try to find a clique of this size
            new_seed = find_max_clique_seed(adj, n)
            if len(new_seed) < restart_seed_size:
                # Just pick random nodes
                new_seed = random.sample(range(n), restart_seed_size)
            
            # Try to solve for this new seed
            paths = find_disjoint_paths(new_seed, time_limit=0.5)
            if paths:
                if len(new_seed) > best_size:
                    current_branches = new_seed
                    current_paths = paths
                    best_size = len(new_seed)
                else:
                    # Even if not larger, accept to explore different basin
                    current_branches = new_seed
                    current_paths = paths
            
            # If we just restarted, loop continues to try adding from this new base.

    # Final Validation and Formatting
    if not current_paths:
        # Fallback
        if edges:
            u, v = edges[0]
            return {'branches': [u, v], 'paths': [[u, v]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    # Ensure types are correct
    final_branches = [int(b) for b in current_branches]
    final_paths = []
    for u, v in combinations(final_branches, 2):
        if (u, v) in current_paths:
            final_paths.append([int(x) for x in current_paths[(u, v)]])
        elif (v, u) in current_paths:
            final_paths.append([int(x) for x in current_paths[(v, u)]])
        else:
            # Should not happen
            final_paths.append([u, v]) # Invalid, but prevents crash. Will fail validation.
            
    # One last check
    if not is_valid_configuration(final_branches, {tuple(p[0], p[-1]): p for p in final_paths}):
         # If invalid, return the known best? 
         # We don't store "best" separately, we mutate "current".
         # If current is invalid, something went wrong.
         # Fallback to seed.
         if edges:
            u, v = edges[0]
            return {'branches': [u, v], 'paths': [[u, v]]}
            
    return {
        'branches': final_branches,
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
