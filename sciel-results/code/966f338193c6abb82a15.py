"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 145.0  # Safety margin for 150s kill

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _apply_2opt(tour, distance_matrix, max_passes=3):
    """
    Apply 2-opt improvement to a list of nodes representing an open path.
    The cost function includes the closing edge back to the start.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    # Copy tour to avoid modifying original
    t = tour[:]
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip the case where we would reverse the entire path
                # which is equivalent to the same cycle in reverse
                if i == 0 and j == n - 1:
                    continue
                    
                # Calculate delta cost
                # Current edges: (t[i], t[i+1]) and (t[j], t[j+1] or t[0] if j==n-1)
                # New edges: (t[i], t[j]) and (t[i+1], t[j+1] or t[0] if j==n-1)
                
                if j == n - 1:
                    # Closing edge involved
                    old_cost = distance_matrix[t[i], t[i+1]] + distance_matrix[t[j], t[0]]
                    new_cost = distance_matrix[t[i], t[j]] + distance_matrix[t[i+1], t[0]]
                else:
                    old_cost = distance_matrix[t[i], t[i+1]] + distance_matrix[t[j], t[j+1]]
                    new_cost = distance_matrix[t[i], t[j]] + distance_matrix[t[i+1], t[j+1]]
                
                if new_cost < old_cost - 1e-8:
                    # Reverse segment t[i+1] to t[j]
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                    improved = True
                    # Break inner loop and restart outer for correctness/simplicity
                    # This is slower but ensures we don't miss improvements in the same pass
                    break
            
            if improved:
                break
        
    return t

def _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node by evaluating each candidate with a fast greedy completion
    and a limited 2-opt refinement.
    """
    best_candidate = -1
    best_score = float('inf')
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Pre-compute distances from current_node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Sort candidates by distance to current node? 
    # Actually, we want to evaluate all, but if time is tight, we might prioritize closer ones.
    # For N=50, evaluating all 50 is fine if the simulation is fast.
    
    for idx, candidate in enumerate(unvisited_nodes):
        # Simulate the rest of the tour using Nearest Neighbor
        # Start with current_tour + [candidate]
        sim_tour = _current_tour + [candidate]
        
        # Remaining nodes are unvisited_nodes excluding candidate
        # To avoid creating a new array with np.delete (which is slow), we can use a boolean mask
        # or just iterate through unvisited_nodes and skip candidate.
        
        # Let's use a list of remaining nodes
        remaining = list(unvisited_nodes)
        remaining.remove(candidate)
        
        curr = candidate
        # Complete the tour using NN
        for _ in range(len(remaining)):
            if not remaining:
                break
            # Find nearest neighbor in remaining
            # This is O(K) per step, so O(K^2) total for simulation
            # K is at most 49. 49^2 = 2401 operations.
            # For 50 candidates, 50 * 2401 = 120,000 operations.
            # Plus 2-opt.
            
            # Optimize: Use numpy for min
            # Convert remaining to array? No, it's a list.
            # Let's stick to simple loop for now, or convert to array once.
            
            # Better: Keep remaining as numpy array and use argmin
            # But we need to remove elements.
            
            # Let's re-implement simulation with numpy arrays for speed
            pass
        
        # Actually, let's rewrite the simulation part to be more efficient
        # We'll create a numpy array for remaining nodes
        remaining_arr = np.array(remaining)
        
        curr = candidate
        while len(remaining_arr) > 0:
            # Find nearest neighbor
            dists = distance_matrix[curr, remaining_arr]
            next_idx = np.argmin(dists)
            next_node = remaining_arr[next_idx]
            sim_tour.append(int(next_node))
            curr = next_node
            # Remove next_node from remaining_arr
            remaining_arr = np.delete(remaining_arr, next_idx)
        
        # Evaluate the simulated tour with 2-opt
        # Check time
        if time.time() - _start_time > _TIME_LIMIT - 10:
            # Fallback to simple distance
            score = dists_from_current[idx]
        else:
            # Apply 2-opt
            optimized_tour = _apply_2opt(sim_tour, distance_matrix, max_passes=2)
            
            # Calculate total closed tour length
            n_tour = len(optimized_tour)
            if n_tour == 0:
                score = 0.0
            elif n_tour == 1:
                score = 0.0
            else:
                total = 0.0
                for i in range(n_tour - 1):
                    total += distance_matrix[optimized_tour[i], optimized_tour[i+1]]
                total += distance_matrix[optimized_tour[-1], optimized_tour[0]]
                score = total
        
        if score < best_score:
            best_score = score
            best_candidate = candidate
            
    return int(best_candidate)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # Use lookahead strategy
    next_node = _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Update global state
    _current_tour.append(next_node)
    
    return next_node

# EVOLVE-BLOCK-END
