import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Filter self-conflicts
    # Node v is valid if 2*v is not a square
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    node_set = set(valid_nodes)
    
    # Build adjacency masks
    # Bit i corresponds to node i.
    adj_mask = [0] * (N + 1) # 1-indexed, adj_mask[v]
    
    for v in valid_nodes:
        for u in valid_nodes:
            if u > v and (v + u) in squares:
                adj_mask[v] |= (1 << u)
                adj_mask[u] |= (1 << v)
                
    initial_mask = 0
    for v in valid_nodes:
        initial_mask |= (1 << v)
        
    best_set = set()
    best_size = 0
    
    # Mod 3 baseline
    mod3_set = set(v for v in range(1, N+1) if v % 3 == 1)
    # Verify mod3
    # a,b in mod3 => a+b = 2 mod 3. Squares mod 3 are 0, 1. 2 is not a square.
    # Diagonal: 2a = 2 mod 3. Not a square.
    # So mod3 is valid.
    best_set = mod3_set
    best_size = len(mod3_set)
    
    def is_valid_set(s):
        """Check if set s is valid (no pair sums to a square, including diagonals)."""
        for a in s:
            for b in s:
                if a <= b and (a + b) in squares:
                    return False
        return True
    
    def local_search(sol_set, max_iters=200):
        """Improve a solution via local search (swap operations)."""
        sol = set(sol_set)
        sol_list = list(sol)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 3.0:
                break
            
            # Try swapping each element in sol with each element not in sol
            for i, in_elem in enumerate(sol_list):
                if time.time() > deadline - 3.0:
                    break
                    
                for out_elem in valid_nodes:
                    if out_elem in sol:
                        continue
                    
                    # Check if swap is beneficial
                    # Remove in_elem, add out_elem
                    # Check validity: out_elem must not conflict with any remaining element
                    # and in_elem's removal must not create issues (it won't since we're removing)
                    
                    # Check if out_elem conflicts with any element in sol except in_elem
                    valid_swap = True
                    for other in sol:
                        if other == in_elem:
                            continue
                        if (other + out_elem) in squares:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        # Perform swap
                        sol.discard(in_elem)
                        sol.add(out_elem)
                        improved = True
                        sol_list = list(sol)
                        break
                if improved:
                    break
        
        return sol
    
    # Bitmask Greedy
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        current_mask = initial_mask
        sol = []
        
        while current_mask:
            min_deg = N + 1
            candidates = []
            
            for v in valid_nodes:
                if current_mask & (1 << v):
                    deg = (adj_mask[v] & current_mask).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
            
            if not candidates:
                break
                
            v = random.choice(candidates)
            sol.append(v)
            current_mask &= ~(adj_mask[v] | (1 << v))
        
        sol_set = set(sol)
        
        # Apply local search
        improved_sol = local_search(sol_set)
        
        if len(improved_sol) > best_size:
            best_size = len(improved_sol)
            best_set = improved_sol
    
    # Final verification and fallback
    if not is_valid_set(best_set):
        # Fallback to mod3 which is guaranteed valid
        best_set = mod3_set
        best_size = len(mod3_set)
    
    return list(best_set)
