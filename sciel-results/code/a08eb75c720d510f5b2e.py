"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_path = None
    best_len = 0
    
    # Precompute neighbors for each vertex
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def solve_snake():
        """
        Uses a backtracking search with a block_count optimization.
        block_count[v] tracks how many vertices in path[0:-1] are at distance 1 from v.
        A vertex v is a valid next step if used[v] is False and block_count[v] == 0.
        """
        path = []
        used = [False] * (1 << N)
        block_count = [0] * (1 << N)
        
        def backtrack(depth):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return False
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
                
            if len(path) == (1 << N):
                return True
                
            cur = path[-1]
            
            # Get valid candidates
            candidates = []
            for v in neighbors[cur]:
                if not used[v] and block_count[v] == 0:
                    candidates.append(v)
            
            if not candidates:
                return False
                
            # Shuffle for randomness
            random.shuffle(candidates)
            
            for v in candidates:
                # Add v
                path.append(v)
                used[v] = True
                
                # Update block_count: neighbors of path[-2] become blocked
                if len(path) >= 2:
                    prev = path[-2]
                    for u in neighbors[prev]:
                        block_count[u] += 1
                        
                # Recurse
                if backtrack(depth + 1):
                    return True
                
                # Backtrack
                if len(path) >= 2:
                    prev = path[-2]
                    for u in neighbors[prev]:
                        block_count[u] -= 1
                        
                used[v] = False
                path.pop()
                
            return False
            
        # Start search
        backtrack(0)
        
    # We can't do full backtracking to depth 512. We need a heuristic approach.
    # The block_count method is fast, so we can do greedy with randomization and limited backtracking.
    
    def greedy_with_backtrack(start_v, max_depth, rng, time_limit):
        path = [start_v]
        used = [False] * (1 << N)
        used[start_v] = True
        block_count = [0] * (1 << N)
        
        # Helper to add a vertex
        def add_vertex(v):
            path.append(v)
            used[v] = True
            if len(path) >= 2:
                prev = path[-2]
                for u in neighbors[prev]:
                    block_count[u] += 1
                    
        def remove_vertex():
            v = path.pop()
            used[v] = False
            if len(path) >= 2:
                prev = path[-2]
                for u in neighbors[prev]:
                    block_count[u] -= 1
            return v
        
        # Greedy phase
        while time.time() < time_limit:
            cur = path[-1]
            candidates = [v for v in neighbors[cur] if not used[v] and block_count[v] == 0]
            
            if not candidates:
                break
                
            rng.shuffle(candidates)
            v = candidates[0]
            add_vertex(v)
            
        # Backtracking phase: try to extend further by backtracking a few steps
        # We will try to improve the path by backtracking up to 'max_backtrack' steps
        # and choosing a different candidate.
        
        # To make this efficient, we will simulate a DFS with a depth limit on the backtrack tree
        # but we only do this if the greedy path is promising.
        
        # A simpler approach: Randomized restarts are often better for this kind of problem.
        # But let's try a simple local search improvement.
        
        # For now, just return the greedy path. The speedup allows many more restarts.
        return path

    # Multiple restarts
    # Since greedy is now O(1) per step, we can do many restarts.
    # Each restart is O(L) where L is the length of the snake.
    # If L is ~100, and we have 140 seconds, we can do a LOT of restarts.
    
    for restart in range(100000):
        if time.time() > deadline - 1.0:
            break
            
        rng = random.Random(restart * 7919 + 1)
        start_v = rng.randint(0, (1 << N) - 1)
        
        # Give each restart a small time budget? No, just run greedy until stuck.
        # Greedy is very fast now.
        path = greedy_with_backtrack(start_v, 512, rng, deadline - 1.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        # If we find a very long snake, we might stop
        if best_len >= 190:
            break
            
    if best_path is None:
        return [0]
        
    return best_path
# EVOLVE-BLOCK-END
