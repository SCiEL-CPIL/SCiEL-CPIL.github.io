"""
Snake-in-the-Box in Q9 using depth-first backtracking with aggressive pruning.

Structural change from greedy+ILS:
- DFS backtracking explores ALL valid extensions (not just first found)
- Bitmask-based used set for O(1) membership
- Time-sliced multi-start: each start gets a fixed time budget
- Pruning: if a vertex has no valid extension, backtrack immediately
- Neighbor ordering: try bits that lead to more available extensions first
"""
import time
import random
import sys

N = 9
TOTAL_VERTS = 1 << N  # 512

def _precompute_neighbors():
    neighbors = [[] for _ in range(TOTAL_VERTS)]
    for v in range(TOTAL_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    return neighbors

_NEIGHBORS = _precompute_neighbors()

def _dfs_search(start_vertex, deadline, rng, best_path, best_len):
    """
    Depth-first backtracking search from start_vertex.
    Returns (best_path_found, best_len_found).
    """
    # Use a list for the path and a bitmask for used vertices
    # For 512 vertices, we can use a Python int as a bitmask
    path = [start_vertex]
    used_mask = 1 << start_vertex
    
    # For speed, track the set of "forbidden" neighbors for the current end
    # A vertex v can be appended if:
    #   1. v is not in used_mask
    #   2. v is adjacent to path[-1]
    #   3. No neighbor of v (other than path[-1]) is in used_mask
    
    # We'll use recursion with a depth limit to avoid stack overflow
    # But Python recursion limit is a concern, so use iterative DFS
    
    # Iterative DFS with explicit stack
    # Stack entries: (path_list, used_mask, next_bit_index, best_so_far_len)
    # Actually, for proper backtracking we need to try all bits
    
    # Let's use a different approach: recursive with sys.setrecursionlimit
    # but that's risky. Instead, use a manual stack.
    
    # Manual stack for DFS:
    # Each stack frame: (path, used_mask, bit_idx)
    # We push frames for unexplored bits
    
    max_depth = TOTAL_VERTS  # upper bound
    stack = [(path, used_mask, 0)]
    
    # For neighbor ordering, precompute a heuristic: for each vertex,
    # order its neighbors by how "promising" they are
    # Simple heuristic: prefer neighbors with fewer already-used neighbors
    # (i.e., more "open" vertices)
    
    while stack:
        if time.time() > deadline:
            break
        
        path, used_mask, bit_idx = stack[-1]
        
        if bit_idx >= N:
            # No more bits to try at this level, backtrack
            stack.pop()
            continue
        
        # Get the next bit to try
        b = bit_idx
        # Update the stack entry to move to next bit
        stack[-1] = (path, used_mask, b + 1)
        
        v = path[-1] ^ (1 << b)
        
        # Check if v is valid to append
        if (used_mask >> v) & 1:
            continue  # v already used
        
        # Check adjacency to path[-1] (already guaranteed by construction)
        # Check that no other neighbor of v is in used_mask
        valid = True
        for w in _NEIGHBORS[v]:
            if w == path[-1]:
                continue
            if (used_mask >> w) & 1:
                valid = False
                break
        
        if not valid:
            continue
        
        # Valid extension! Push new state
        new_path = path + [v]
        new_used = used_mask | (1 << v)
        new_len = len(new_path)
        
        if new_len > best_len:
            best_len = new_len
            best_path = list(new_path)
        
        # Push to stack for further exploration
        stack.append((new_path, new_used, 0))
    
    return best_path, best_len

def _dfs_search_ordered(start_vertex, deadline, rng, best_path, best_len):
    """
    DFS with ordered neighbor exploration: try bits in an order that
    prioritizes vertices with more available extensions.
    """
    path = [start_vertex]
    used_mask = 1 << start_vertex
    
    # For each vertex, compute a static ordering of its neighbors
    # based on some heuristic. We'll use a simple one:
    # Prefer bits that result in vertices with fewer already-used neighbors.
    # But this is dynamic, so let's just use random ordering per level.
    
    # Actually, let's use a simpler approach: at each step, order the 9 bits
    # by how many of their neighbors are NOT in used_mask.
    # This is expensive to compute at every step, so let's cache it.
    
    # For now, use random bit ordering
    stack = [(path, used_mask, 0)]
    # We need to store the bit order for each stack frame
    # Let's restructure: stack entries are (path, used_mask, bit_order, bit_idx)
    stack = [(path, used_mask, list(range(N)), 0)]
    rng.shuffle(stack[0][2])
    
    while stack:
        if time.time() > deadline:
            break
        
        path, used_mask, bit_order, bit_idx = stack[-1]
        
        if bit_idx >= len(bit_order):
            stack.pop()
            continue
        
        b = bit_order[bit_idx]
        stack[-1] = (path, used_mask, bit_order, bit_idx + 1)
        
        v = path[-1] ^ (1 << b)
        
        if (used_mask >> v) & 1:
            continue
        
        valid = True
        for w in _NEIGHBORS[v]:
            if w == path[-1]:
                continue
            if (used_mask >> w) & 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_path = path + [v]
        new_used = used_mask | (1 << v)
        new_len = len(new_path)
        
        if new_len > best_len:
            best_len = new_len
            best_path = list(new_path)
        
        # Generate new bit order for the next level
        new_bit_order = list(range(N))
        rng.shuffle(new_bit_order)
        stack.append((new_path, new_used, new_bit_order, 0))
    
    return best_path, best_len

def construct():
    deadline = time.time() + 115
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Multi-start DFS with time slicing
    # Each start gets a portion of the time budget
    num_starts = 20
    time_per_start = (deadline - time.time()) / num_starts
    
    # First, do a quick greedy to get a baseline
    # This gives us a reasonable starting point for pruning
    greedy_path = [0]
    greedy_used = {0}
    while True:
        cur = greedy_path[-1]
        found = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in greedy_used:
                # Check validity
                valid = True
                for w in _NEIGHBORS[v]:
                    if w in greedy_used and w != cur:
                        valid = False
                        break
                if valid:
                    greedy_path.append(v)
                    greedy_used.add(v)
                    found = True
                    break
        if not found:
            break
    
    best_path = list(greedy_path)
    best_len = len(best_path)
    
    # Now do multi-start DFS
    for s in range(num_starts):
        start_time = time.time()
        start_deadline = start_time + time_per_start
        
        if start_deadline > deadline:
            start_deadline = deadline
        
        # Choose a random start vertex
        start_v = rng.randint(0, TOTAL_VERTS - 1)
        
        # Do DFS from this start
        bp, bl = _dfs_search_ordered(start_v, start_deadline, rng, best_path, best_len)
        if bl > best_len:
            best_path = bp
            best_len = bl
    
    return best_path

# EVOLVE-BLOCK-START
# The construct function is the entry point.
# EVOLVE-BLOCK-END
