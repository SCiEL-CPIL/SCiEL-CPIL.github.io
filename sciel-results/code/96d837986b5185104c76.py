import random
import time

N = 9
V = 1 << N
MAX_TIME = 110

# Precompute neighbors bitmask for each vertex
NEIGHBOR_MASKS = [0] * V
for v in range(V):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASKS[v] = mask

def construct():
    best_path = []
    best_len = 0
    
    start_time = time.time()
    
    # Backtracking function
    # path: list of vertices
    # used_mask: bitmask of used vertices
    # bad_mask: bitmask of vertices that are adjacent to path[:-1]
    def backtrack(path, used_mask, bad_mask):
        nonlocal best_path, best_len
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        if time.time() - start_time > MAX_TIME:
            return
            
        cur = path[-1]
        
        # Get candidates: neighbors of cur that are not used and not in bad_mask
        candidates_mask = NEIGHBOR_MASKS[cur] & ~used_mask & ~bad_mask
        
        # Convert mask to list of vertices
        candidates = []
        m = candidates_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
            
        # Shuffle for randomization to explore different branches
        random.shuffle(candidates)
        
        for v in candidates:
            # Update state
            new_used = used_mask | (1 << v)
            # New bad mask adds neighbors of the vertex we are leaving (cur)
            # The bad mask for the next step is based on path[:-1] of the new path,
            # which is the current path.
            # So we add neighbors of cur to bad_mask.
            new_bad = bad_mask | NEIGHBOR_MASKS[cur]
            
            # Check if v is in new_bad? 
            # v is a neighbor of cur. cur is in path. 
            # In the new path, cur is path[-2]. 
            # The bad mask for the *next* step (after v) should prevent neighbors of cur.
            # But we need to check if v is valid *now*.
            # v was checked against the *old* bad_mask.
            # The old bad_mask contained neighbors of path[:-2].
            # So v is not adjacent to path[:-2].
            # v is adjacent to path[-1] (cur) by construction.
            # So v is valid.
            
            path.append(v)
            backtrack(path, new_used, new_bad)
            path.pop()
            
            if time.time() - start_time > MAX_TIME:
                return

    # Run multiple randomized backtracking attempts
    # We can start from 0 or other starting points
    # To maximize diversity, we can shuffle the bit order for neighbor generation? 
    # No, the graph is fixed. We just shuffle the candidate order.
    
    # Initial greedy to get a baseline quickly? 
    # The backtracking will find short paths very fast.
    # Let's just run backtracking.
    
    # To avoid getting stuck in a local optimum of the search tree, 
    # we can limit the depth of backtracking in some runs or just let it run.
    # Given 120s, we can do many random restarts.
    
    # Start vertex 0
    # We can also try different start vertices, but 0 is symmetric.
    
    # To improve performance, we can add a "greedy phase" first to get a decent length,
    # then backtrack from there? 
    # Or just pure randomized backtracking.
    
    # Let's do a mix: 
    # 1. Run a quick greedy to get a path of length ~100.
    # 2. Use that as a seed for backtracking? 
    # Backtracking from a long seed is hard because the state is deep.
    # Better to start from scratch with randomized backtracking.
    
    # Optimization: If we find a path of length > 150, we can increase the time limit for subsequent searches?
    # No, just keep searching.
    
    # Random seed for reproducibility within a run? 
    # random.seed() is called by evaluator? No, we should ensure variety.
    
    # To ensure we explore different parts of the search space, 
    # we can vary the shuffle seed or just rely on random.shuffle.
    
    # Let's run backtracking.
    # To prevent infinite loops or too slow execution, the time check is inside.
    
    # One issue: Python recursion limit.
    # Max depth is 512. Default is 1000. Safe.
    
    # Start
    path = [0]
    used_mask = 1
    # bad_mask for next step (neighbors of path[:-1] which is empty) is 0
    bad_mask = 0
    
    backtrack(path, used_mask, bad_mask)
    
    return best_path
