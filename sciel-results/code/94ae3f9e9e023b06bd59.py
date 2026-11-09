import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    N = 256
    
    # Precompute squares up to 2*N
    max_sum = 2 * N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter nodes that violate the diagonal condition (a=a => 2a is square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # 2. Build conflict graph using bitmasks
    # adj_mask[i] is a bitmask of nodes j such that valid_nodes[i] + valid_nodes[j] is a square
    adj_mask = [0] * M
    for i in range(M):
        vi = valid_nodes[i]
        mask = 0
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                mask |= (1 << j)
                adj_mask[j] |= (1 << i)
        adj_mask[i] = mask
        
    # 3. Branch and Bound for Maximum Independent Set
    # We want to maximize the number of nodes selected such that no two are adjacent.
    
    best_set = []
    best_size = 0
    
    # Heuristic for initial lower bound: Greedy
    # Pick nodes with lowest degree first? Or just simple greedy.
    # Let's do a quick greedy to get a good initial bound.
    greedy_available = (1 << M) - 1
    greedy_set = []
    while greedy_available:
        # Find node with min degree in remaining graph
        min_deg = M + 1
        min_node = -1
        # Optimization: Iterate through set bits
        temp = greedy_available
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            # Degree of idx in greedy_available
            # Neighbors of idx are adj_mask[idx]
            # Common neighbors in greedy_available: adj_mask[idx] & greedy_available
            deg = (adj_mask[idx] & greedy_available).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_node = idx
            temp ^= lsb
            
        if min_node == -1: break
        greedy_set.append(min_node)
        # Remove min_node and its neighbors from available
        greedy_available &= ~(1 << min_node)
        greedy_available &= ~adj_mask[min_node]
        
    best_size = len(greedy_set)
    best_set = list(greedy_set)
    
    # B&B Solver
    # To avoid recursion depth issues, we might need to be careful, but M is small (~150-200).
    # However, Python recursion limit might be an issue if the depth is large.
    # We can increase recursion limit or use iterative deepening if needed, but usually B&B depth is bounded by the solution size.
    sys.setrecursionlimit(10000)
    
    def solve(available, current_set, current_size):
        nonlocal best_size, best_set
        
        # Pruning
        if current_size + available.bit_count() <= best_size:
            return
            
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Heuristic: Pick a node to branch on.
        # Common heuristic: Pick node with max degree in 'available' to prune faster.
        # Or pick node with min degree to keep options open?
        # For MIS, picking a high-degree node is often better because it eliminates many options.
        
        # Find node with max degree in available
        max_deg = -1
        branch_node = -1
        temp = available
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            # Degree in available
            deg = (adj_mask[idx] & available).bit_count()
            if deg > max_deg:
                max_deg = deg
                branch_node = idx
            temp ^= lsb
            
        if branch_node == -1:
             if current_size > best_size:
                 best_size = current_size
                 best_set = current_set[:]
             return
                 
        # Branch 1: Include branch_node
        # New available = available without branch_node and its neighbors
        new_available_include = available & ~(1 << branch_node) & ~adj_mask[branch_node]
        current_set.append(branch_node)
        solve(new_available_include, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: Exclude branch_node
        # New available = available without branch_node
        new_available_exclude = available & ~(1 << branch_node)
        solve(new_available_exclude, current_set, current_size)
        
        # Time check
        if time.time() > deadline - 2:
            return # Early exit signal? Hard to do cleanly in recursion without exceptions or flags.
            # Let's use a global flag.
        
    # We need a way to stop early if time runs out.
    # Using a global flag is cleaner.
    stop_search = False
    
    def solve_bounded(available, current_set, current_size):
        nonlocal best_size, best_set, stop_search
        if stop_search:
            return
        
        if current_size + available.bit_count() <= best_size:
            return
            
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Heuristic: Pick node with max degree in available
        max_deg = -1
        branch_node = -1
        temp = available
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (adj_mask[idx] & available).bit_count()
            if deg > max_deg:
                max_deg = deg
                branch_node = idx
            temp ^= lsb
            
        if branch_node == -1:
             if current_size > best_size:
                 best_size = current_size
                 best_set = current_set[:]
             return
                 
        # Branch 1: Include
        new_available_include = available & ~(1 << branch_node) & ~adj_mask[branch_node]
        current_set.append(branch_node)
        solve_bounded(new_available_include, current_set, current_size + 1)
        current_set.pop()
        
        if stop_search:
            return
            
        # Branch 2: Exclude
        new_available_exclude = available & ~(1 << branch_node)
        solve_bounded(new_available_exclude, current_set, current_size)
        
        if time.time() > deadline - 2:
            stop_search = True
            return

    # Run B&B
    solve_bounded((1 << M) - 1, [], 0)
    
    # Convert indices back to values
    result = [valid_nodes[i] for i in best_set]
    
    # If result is empty (should not happen), fallback
    if not result:
        return [1]
        
    return result
