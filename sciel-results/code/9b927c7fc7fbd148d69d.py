import time
import random
import sys

N = 9
MAX_VERTS = 2 ** N

# Precompute neighbors for all vertices
NEIGHBORS = [None] * MAX_VERTS
for v in range(MAX_VERTS):
    nbrs = []
    for b in range(N):
        nbrs.append(v ^ (1 << b))
    NEIGHBORS[v] = nbrs

def construct():
    """
    Construct the longest induced path (snake) in Q_9 within the time budget.
    Returns a list of distinct integers representing the vertices in order.
    """
    deadline = time.time() + 115  # Safety margin
    
    best_path = []
    best_len = 0
    
    # Helper to check if a vertex is valid to add to a path
    # path: list of vertices
    # v: candidate vertex
    # side: 0 for head (insert at front), 1 for tail (append at end)
    # We need to check against the "far" end and all intermediate vertices.
    # Actually, for an induced path, v must not be adjacent to ANY vertex in the path 
    # except the one it connects to.
    
    def is_valid_addition(path, v, connect_idx):
        """
        Check if v can be added to path at position connect_idx (either 0 or len(path)-1).
        v must be adjacent to path[connect_idx].
        v must NOT be adjacent to any other vertex in path.
        """
        # Check non-adjacency to all other vertices
        # Optimization: Only check vertices that are NOT the connection point.
        # Since v is a neighbor of path[connect_idx], we know bit_count(v ^ path[connect_idx]) == 1.
        # We need to ensure bit_count(v ^ p) >= 2 for all p in path where p != path[connect_idx].
        
        # For performance, we can maintain a set of used vertices, but checking adjacency 
        # to all used vertices is O(L). 
        # However, we can precompute a bitmask or use a faster check.
        # Given N=9, L~200, O(L) check is ~200 operations.
        
        # Let's just do the loop.
        for i, p in enumerate(path):
            if i == connect_idx:
                continue
            if (v ^ p).bit_count() == 1:
                return False
        return True

    def dfs_search(start_v, max_depth, rng):
        """
        Perform a DFS search starting from start_v, trying to extend the path.
        We will maintain a path and a set of used vertices.
        To speed up, we will try to grow from the tail first, and if stuck, try the head.
        Actually, a full DFS on Q9 is too big. We will use a heuristic-guided DFS 
        with limited backtracking (beam search or iterative deepening with pruning).
        
        Alternative: Use a "greedy with backtracking" approach.
        1. Start with a path [start_v].
        2. Try to extend the tail. If multiple options, pick one (randomly or by heuristic).
        3. If stuck, backtrack to the last decision point and try a different option.
        4. If we reach a certain length or time limit, return the best path found.
        """
        # We will implement a recursive-like search using a stack to allow backtracking.
        # State: (path, used_set, next_decision_index)
        # This is complex. Let's try a simpler approach: 
        # "Greedy with randomized restarts and local repair" is what the current program does.
        # The prompt asks for a DIFFERENT strategy.
        
        # Let's try: **Iterative Deepening DFS with Conflict Pruning**
        # But Q9 has 512 vertices. A path of length 200 is very long.
        # Standard DFS will get stuck quickly.
        
        # New Idea: **Simulated Annealing / Local Search on the Path**
        # 1. Generate a valid snake using a fast greedy method (like the current one but simpler).
        # 2. Try to improve it by "swapping" or "rotating" segments of the path to free up space.
        # This is hard to implement correctly for induced paths.
        
        # Let's stick to a **more aggressive greedy with lookahead** but different from Warnsdorff.
        # Current: Warnsdorff (min options).
        # New: **Maximize Distance / Spread**
        # Instead of picking the neighbor with fewest options, pick the one that is "furthest" 
        # from the center of mass of the path, or simply the one that maximizes the minimum distance 
        # to other path vertices? No, that's expensive.
        
        # Let's try **Randomized Restart with a different heuristic: "Avoid Clustering"**
        # Score = - (number of neighbors of candidate that are already in the path) 
        # Wait, if a neighbor is in the path, it's invalid unless it's the connection point.
        # So all valid candidates have 0 neighbors in the path (except the connection point).
        # So that score is always 0.
        
        # Let's look at the neighbors of the NEIGHBORS of the candidate.
        # If I add v, how many of v's other neighbors are "blocked" by the path?
        # If many are blocked, v is in a "dead end" pocket.
        # If few are blocked, v is in an "open" area.
        # Warnsdorff says: pick the one with FEWEST open neighbors (to close off dead ends first).
        # Let's try the opposite: **Maximize Open Neighbors** (to keep the path in open areas).
        # Or a mix: **Adaptive Heuristic**
        
        # Actually, the most promising different strategy for hypercubes is often **Hamiltonian Path Construction**.
        # Q_n has a Hamiltonian cycle. A snake is a subgraph of a Hamiltonian path.
        # If we can find a Hamiltonian path, we can take a prefix of it.
        # But the induced path constraint is strong.
        # In Q_9, a Hamiltonian path of length 511 is likely NOT an induced path.
        # (In a Hamiltonian path, v_i and v_{i+2} might be adjacent, which violates the induced path condition).
        
        # So we must stick to induced paths.
        
        # Let's try **Bidirectional Growth with a "Lookahead" Depth of 2**.
        # When choosing a move, simulate the next move and see if it leads to a dead end.
        
        pass # This function is a placeholder for the actual logic below

    # We will implement a **Beam Search** style approach.
    # Maintain a set of "active" paths (states).
    # At each step, expand each active path by its valid neighbors.
    # Keep only the top K paths based on a score (e.g., length, or potential to grow).
    # This allows exploring multiple possibilities in parallel and backtracking implicitly.
    
    K = 20  # Beam width
    
    def beam_search(start_v, max_steps, rng):
        active_paths = [[start_v]]
        active_sets = [{start_v}]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            new_active_paths = []
            new_active_sets = []
            
            for path, pset in zip(active_paths, active_sets):
                # Try to extend head
                head = path[0]
                for nbr in NEIGHBORS[head]:
                    if nbr in pset:
                        continue
                    if is_valid_addition(path, nbr, 0):
                        new_path = [nbr] + path
                        new_set = pset | {nbr}
                        new_active_paths.append(new_path)
                        new_active_sets.append(new_set)
                        
                # Try to extend tail
                tail = path[-1]
                for nbr in NEIGHBORS[tail]:
                    if nbr in pset:
                        continue
                    if is_valid_addition(path, nbr, len(path)-1):
                        new_path = path + [nbr]
                        new_set = pset | {nbr}
                        new_active_paths.append(new_path)
                        new_active_sets.append(new_set)
            
            if not new_active_paths:
                break
                
            # Score and select top K
            # Score: Length is the primary goal. But we want to keep paths that have "potential".
            # All paths in the beam have the same length (step+1).
            # So we need a heuristic for "potential".
            # Potential = Number of valid extensions available in the next step?
            # This is expensive.
            # Let's just keep the first K or random K?
            # Random K might lose good paths.
            # Let's use a score: Sum of "openness" of the ends.
            
            scored_paths = []
            for i, path in enumerate(new_active_paths):
                head = path[0]
                tail = path[-1]
                pset = new_active_sets[i]
                
                head_open = 0
                for nbr in NEIGHBORS[head]:
                    if nbr not in pset:
                        # Check validity quickly? No, just count non-used.
                        head_open += 1
                        
                tail_open = 0
                for nbr in NEIGHBORS[tail]:
                    if nbr not in pset:
                        tail_open += 1
                        
                score = head_open + tail_open
                # Add some randomness
                score += rng.random()
                scored_paths.append((score, i))
                
            scored_paths.sort(key=lambda x: x[0], reverse=True)
            
            next_indices = [idx for _, idx in scored_paths[:K]]
            
            # Shuffle to avoid bias
            rng.shuffle(next_indices)
            
            active_paths = [new_active_paths[i] for i in next_indices]
            active_sets = [new_active_sets[i] for i in next_indices]
            
            # Update best
            current_len = len(new_active_paths[0])
            if current_len > best_len:
                best_len = current_len
                # We need to return the best path found so far.
                # Since all paths in the beam have the same length, pick the first one.
                # But we should keep the best one globally.
                # Let's store the best path in a non-local variable or return it.
                # For now, we'll update a global best.
                
        return active_paths[0] if active_paths else []

    # Main loop: multiple restarts with Beam Search
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(i * 12345 + 67890)
        
        # Start with a random vertex
        start_v = rng.randint(0, MAX_VERTS - 1)
        
        # Run beam search
        # Max steps: 512 (max possible)
        path = beam_search(start_v, 512, rng)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Fallback
    if best_path is None or len(best_path) == 0:
        best_path = [0]
        
    return best_path

# EVOLVE-BLOCK-END
