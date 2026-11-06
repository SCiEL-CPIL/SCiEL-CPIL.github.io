import time
import random
import sys
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0  # 5s safety margin before 120s limit
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = []
    best_len = 0
    
    def compute_block_counts(used_mask, path):
        """
        Compute for each vertex v (not in path) how many neighbors of v are in the path.
        A vertex v can be added to the path end if it has exactly 1 neighbor in the path (the one we are connecting to) 
        and 0 other neighbors in the path. 
        Actually, the condition for extending the path at an endpoint 'end' to 'n' is:
        1. n is not in path.
        2. n is a neighbor of 'end'.
        3. n has no other neighbors in the path.
        
        So, block_count[v] = number of neighbors of v that are in the path.
        If we want to add v to the path, we need block_count[v] == 1 (only the endpoint is connected to it).
        """
        block_count = [0] * MAX_V
        for v in path:
            for n in neighbors[v]:
                if not (used_mask >> n) & 1:
                    block_count[n] += 1
        return block_count

    def can_extend_from_end(path, used_mask, block_count, end_idx):
        """
        Check if we can extend path from given end (0 for start, 1 for end).
        Returns the vertex to add, or -1.
        """
        cur = path[-1] if end_idx == 1 else path[0]
        for n in neighbors[cur]:
            if not (used_mask >> n) & 1 and block_count[n] == 1:
                return n
        return -1

    def extend_path(path):
        """
        Try to extend path from both ends repeatedly until no more extensions are possible.
        """
        if not path:
            return path
        
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
        block_count = compute_block_counts(used_mask, path)
        
        extended = True
        while extended:
            extended = False
            
            # Try extending from end
            n = can_extend_from_end(path, used_mask, block_count, 1)
            if n >= 0:
                path.append(n)
                used_mask |= (1 << n)
                for nn in neighbors[n]:
                    if not (used_mask >> nn) & 1:
                        block_count[nn] += 1
                extended = True
                continue
            
            # Try extending from beginning
            n = can_extend_from_end(path, used_mask, block_count, 0)
            if n >= 0:
                path.insert(0, n)
                used_mask |= (1 << n)
                for nn in neighbors[n]:
                    if not (used_mask >> nn) & 1:
                        block_count[nn] += 1
                extended = True
        
        return path

    def backtrack_search(start, rng, max_depth=None):
        """
        Backtracking search with Warnsdorff's heuristic.
        """
        path = [start]
        used_mask = 1 << start
        block_count = [0] * MAX_V
        for n in neighbors[start]:
            block_count[n] += 1
        
        # We use a recursive structure but manage depth manually to avoid recursion limits
        # and to allow early termination based on time.
        
        # To avoid deep recursion overhead and stack limits, we use an iterative approach 
        # or a careful recursive one. Given N=9, max depth is 512. Recursion limit in Python 
        # is usually 1000, so it's safe, but we must be careful with time.
        
        # Let's use a standard recursive backtracking with heuristic sorting.
        
        def dfs(current_path, current_used_mask, current_block_count):
            nonlocal best_path, best_len, deadline
            
            if time.time() > deadline:
                return False # Signal to stop searching
            
            if len(current_path) > best_len:
                best_len = len(current_path)
                best_path = current_path[:]
                
            # If we reached max possible length (unlikely to reach 512, but good to check)
            if len(current_path) == MAX_V:
                return True
            
            cur = current_path[-1]
            
            # Find valid candidates
            candidates = []
            for n in neighbors[cur]:
                if not (current_used_mask >> n) & 1 and current_block_count[n] == 1:
                    # Calculate heuristic: number of available moves from n
                    # Available moves are neighbors of n that are not used and have block_count 0
                    # relative to the new path if we add n.
                    # Wait, if we add n, block_count of n's neighbors increases.
                    # The "degree" for Warnsdorff is the number of neighbors of n 
                    # that could potentially be added *after* n.
                    # A neighbor nn of n is a valid next step if:
                    # 1. nn is not used.
                    # 2. nn is not the current vertex (cur).
                    # 3. nn has no other neighbors in the path EXCEPT n.
                    #    Currently, block_count[nn] counts neighbors in current_path.
                    #    If we add n, block_count[nn] becomes current_block_count[nn] + (1 if cur==n? no, n is new).
                    #    Actually, if we add n, then for any neighbor nn of n:
                    #    new_block_count[nn] = current_block_count[nn] + 1 (because n is added)
                    #    Wait, this is only true if n was not in the path. It isn't.
                    #    So, for nn to be a valid next step after n:
                    #    new_block_count[nn] must be 1.
                    #    This means current_block_count[nn] + 1 == 1 => current_block_count[nn] == 0.
                    #    Also nn must not be used.
                    
                    deg = 0
                    for nn in neighbors[n]:
                        if not (current_used_mask >> nn) & 1 and nn != cur:
                            if current_block_count[nn] == 0:
                                deg += 1
                    candidates.append((deg, n))
            
            if not candidates:
                return False
            
            # Sort by degree (Warnsdorff's rule: prefer lower degree)
            candidates.sort(key=lambda x: x[0])
            
            # Randomize ties to explore different branches
            # Group by degree
            # For simplicity, we can just shuffle candidates with the same minimum degree
            # Or just pick from the top k.
            
            min_deg = candidates[0][0]
            min_candidates = [c for c in candidates if c[0] == min_deg]
            
            # Randomly select from the best candidates
            rng.shuffle(min_candidates)
            
            # Try all min candidates in random order
            for _, n in min_candidates:
                # Extend path
                new_path = current_path + [n]
                new_used_mask = current_used_mask | (1 << n)
                
                # Update block counts
                # We need to copy the block count array or update and revert.
                # Copying is O(512) which is fast.
                new_block_count = current_block_count[:]
                
                for nn in neighbors[n]:
                    if not (new_used_mask >> nn) & 1:
                        new_block_count[nn] += 1
                
                # Recurse
                if dfs(new_path, new_used_mask, new_block_count):
                    return True # Stop if we hit time limit or found perfect? 
                                # We don't stop on "found best" because we want to keep searching 
                                # for longer ones until time runs out.
                                # But if time is up, we return True to unwind quickly.
            
            return False

        # Start search
        # To avoid deep recursion issues or to control it, we just call dfs.
        # Python recursion limit is usually enough for depth 512.
        # However, backtracking can be very deep. 
        # Let's increase recursion limit just in case.
        sys.setrecursionlimit(10000)
        
        dfs(path, used_mask, block_count)
        
        return best_path

    # Main loop: Multiple restarts
    random.seed(42)
    rng = random.Random(42)
    
    # Generate diverse starting vertices
    start_vertices = list(range(MAX_V))
    rng.shuffle(start_vertices)
    
    # Strategy: 
    # 1. Run a few fast greedy constructions to get a baseline.
    # 2. Run backtracking searches with random starts.
    
    # Greedy baseline
    def greedy_construct(start, rng):
        path = [start]
        used_mask = 1 << start
        block_count = [0] * MAX_V
        for n in neighbors[start]:
            block_count[n] += 1
        
        while True:
            cur = path[-1]
            candidates = []
            for n in neighbors[cur]:
                if not (used_mask >> n) & 1 and block_count[n] == 1:
                    deg = 0
                    for nn in neighbors[n]:
                        if not (used_mask >> nn) & 1 and nn != cur:
                            if block_count[nn] == 0:
                                deg += 1
                    candidates.append((deg, n))
            
            if not candidates:
                break
            
            candidates.sort(key=lambda x: x[0])
            
            # Randomized choice from top candidates
            k = min(3, len(candidates))
            idx = rng.randint(0, k - 1)
            chosen = candidates[idx][1]
            
            path.append(chosen)
            used_mask |= (1 << chosen)
            for nn in neighbors[chosen]:
                block_count[nn] += 1
        
        return path

    # Run some greedy restarts first to get a good seed
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        start = rng.choice(start_vertices)
        path = greedy_construct(start, rng)
        path = extend_path(path)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    # Now run backtracking searches
    # We can't run infinite backtracking. We need to limit depth or rely on time check.
    # The dfs function checks time. 
    # We will run backtracking for a limited number of restarts.
    
    num_bt_restarts = 50
    for i in range(num_bt_restarts):
        if time.time() > deadline - 5:
            break
            
        start = start_vertices[i % len(start_vertices)]
        # We can also try to start from the end of the current best path to extend it?
        # No, backtracking builds from scratch.
        
        # To make backtracking more effective, we can restrict the search depth 
        # or use a "beam search" style, but standard DFS with Warnsdorff is a good bet.
        
        # We will run the backtracking search. 
        # Note: The dfs function updates best_path globally.
        
        # To prevent a single bad start from consuming all time, 
        # we could add a node limit, but time check is usually sufficient 
        # if the heuristic is good (it prunes a lot).
        
        backtrack_search(start, rng)
        
        # After backtracking, try to extend the best path found
        if best_path:
            extended = extend_path(best_path[:])
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = extended

    # Final extension pass
    if best_path:
        final_path = extend_path(best_path[:])
        if len(final_path) > best_len:
            best_path = final_path
            best_len = len(best_path)
            
    if not best_path:
        return [0]
        
    return best_path

# Ensure the function is named construct as required
# The code above defines construct()

if __name__ == "__main__":
    # Simple test
    path = construct()
    print(f"Length: {len(path)}")
    # Verify validity
    valid = True
    for i in range(len(path)-1):
        if bin(path[i] ^ path[i+1]).count('1') != 1:
            valid = False
            break
    if valid:
        for i in range(len(path)):
            for j in range(i+2, len(path)):
                if bin(path[i] ^ path[j]).count('1') < 2:
                    valid = False
                    break
            if not valid:
                break
    print(f"Valid: {valid}")
