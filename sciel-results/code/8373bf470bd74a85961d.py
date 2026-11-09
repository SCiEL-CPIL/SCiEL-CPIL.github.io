import numpy as np
import random
import time
import math

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Precompute Conflict Masks
    # conflict_mask[i] is a bitmask of all nodes j such that (i-j) % m is in forbidden
    conflict_masks = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        conflict_masks[i] = mask

    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')

    # Helper to check if adding node `node` to `current_mask` is valid
    def can_add(current_mask, node):
        if (current_mask >> node) & 1:
            return False
        return (current_mask & conflict_masks[node]) == 0

    # Precompute list of nodes
    nodes = list(range(m))
    
    # Simulated Annealing Parameters
    max_time = 590 # Leave 10s for safety
    start_time = time.time()
    
    # Initial Solution: Greedy
    random.shuffle(nodes)
    current_mask = 0
    for node in nodes:
        if can_add(current_mask, node):
            current_mask |= (1 << node)
    
    best_mask = current_mask
    best_size = popcount(current_mask)
    
    # Temperature Schedule
    initial_temp = 1.0
    cooling_rate = 0.9999
    min_temp = 1e-6
    
    temp = initial_temp
    
    # Precompute adjacency list for faster neighbor checks if needed, 
    # but bitmask operations are fast enough for m=205.
    
    while time.time() - start_time < max_time and temp > min_temp:
        
        # Generate a candidate move
        # Strategy: 
        # 1. Pick a random node from current set to remove (or none)
        # 2. Pick a random node NOT in current set to add (or none)
        # To make it smarter:
        # Identify "blocked" nodes: nodes not in set that conflict with set.
        # Identify "removable" nodes: nodes in set that are "conflicting" with blocked nodes.
        
        # For simplicity and speed in SA, we can just do random 1-swap or 1-add/1-remove.
        # Let's try a weighted random move.
        
        current_size = popcount(current_mask)
        
        # Create list of selected and unselected
        # To avoid O(N) list creation every step, we can just pick random indices and check.
        # But N=205 is small.
        
        move_type = random.random()
        
        new_mask = current_mask
        delta_size = 0
        
        if move_type < 0.3:
            # Try to add a random node
            while True:
                node = random.randint(0, m - 1)
                if not (current_mask >> node) & 1:
                    break
            if can_add(current_mask, node):
                new_mask = current_mask | (1 << node)
                delta_size = 1
            else:
                # Invalid move, skip or treat as penalty? 
                # In SA, we usually only consider valid states. 
                # If we can't add, we might do a swap.
                # Let's force a swap if add fails.
                pass # Fall through to swap logic if add wasn't successful
        
        if delta_size == 0:
            # Try a swap: remove one, add one
            # Pick a random selected node
            while True:
                s_node = random.randint(0, m - 1)
                if (current_mask >> s_node) & 1:
                    break
            
            # Pick a random unselected node
            while True:
                u_node = random.randint(0, m - 1)
                if not (current_mask >> u_node) & 1:
                    break
            
            # Remove s_node
            temp_mask = current_mask & ~(1 << s_node)
            
            # Check if u_node can be added to temp_mask
            if can_add(temp_mask, u_node):
                new_mask = temp_mask | (1 << u_node)
                delta_size = 0 # Size stays same
            else:
                # Invalid swap, try another? Or just skip this iteration?
                # To keep SA moving, let's just pick a different random pair or skip.
                # Skipping is safe but slows down. 
                # Let's try a "kick" instead: remove 2, add 2?
                # Or just accept that this random move was invalid and try again next iteration.
                # For efficiency, let's just continue the loop.
                temp *= cooling_rate
                continue
        
        # Evaluate new state
        new_size = current_size + delta_size
        
        # Acceptance probability
        if new_size >= current_size:
            # Better or equal, always accept (or accept with prob 1 if better)
            current_mask = new_mask
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_mask = current_mask
        else:
            # Worse, accept with probability exp(delta / temp)
            # delta is negative
            delta = new_size - current_size # -1
            if random.random() < math.exp(delta / temp):
                current_mask = new_mask
                current_size = new_size
        
        temp *= cooling_rate
        
        # Periodic restart or diversification?
        # If stuck for a long time, we might want to reset temperature or do a kick.
        # For now, simple cooling.
        
        # Optimization: If we have been at the same size for a while, 
        # we can increase cooling rate or do a kick.
        # Let's add a simple kick mechanism if no improvement for 1000 steps.
        # (This requires tracking steps, which adds overhead. 
        # Given m=205, simple SA might be enough. 
        # Let's stick to standard SA for now.)
        
    # Convert best_mask to array
    result = []
    for i in range(m):
        if (best_mask >> i) & 1:
            result.append(i)
    
    return np.array(result, dtype=np.int64)
