"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 140.0  # Safety margin for 150s kill

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _select_node_geometric(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a fast geometric heuristic:
    1. Distance to current node (continuity)
    2. Distance to destination node (global direction)
    3. Angle/deflection penalty (smoothness)
    
    This is O(N) per candidate, O(N^2) total per step.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get the last 2 nodes of the current tour to calculate deflection
    # If tour is just [start], we can't calculate deflection, so weight it 0
    if len(_current_tour) >= 2:
        prev_node = _current_tour[-2]
        curr_node = _current_tour[-1]
        
        # Vectorized distances
        d_curr = distance_matrix[curr_node, unvisited_nodes]
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # Deflection penalty:
        # We want to minimize the angle change.
        # Cosine of angle between (curr - prev) and (next - curr).
        # Actually, simpler: minimize d(prev, next) - (d(prev, curr) + d(curr, next))? No.
        # A common heuristic is to minimize the "turning cost".
        # Let's use a simple proxy: d(prev, next) is large if we turn sharply or go far.
        # But d(prev, next) is not the edge cost.
        # Let's stick to a weighted sum of distances.
        
        # Heuristic:
        # Score = w1 * d(curr, next) + w2 * d(next, dest)
        # We can add a small penalty for large jumps if we are far from dest?
        # For now, a 50/50 split between local and global is a strong baseline.
        
        # Let's try to be a bit smarter.
        # If we are close to destination, prioritize d(next, dest).
        # If we are far, prioritize d(curr, next) to avoid long initial jumps?
        # Actually, for TSP, "nearest neighbor" is bad because it gets stuck.
        # "Farthest" is bad.
        # A good heuristic is the "Or-opt" style: minimize the increase in tour length.
        # But we don't have the full tour.
        
        # Let's use a weighted score:
        # score = 0.7 * d_curr + 0.3 * d_dest
        # This biases towards moving locally but keeps an eye on the destination.
        
        scores = 0.7 * d_curr + 0.3 * d_dest
    else:
        # First step: just pick the one that balances local and global
        d_curr = distance_matrix[current_node, unvisited_nodes]
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        scores = 0.5 * d_curr + 0.5 * d_dest

    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

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
    
    # Use geometric heuristic
    next_node = _select_node_geometric(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Update global state
    _current_tour.append(next_node)
    
    return next_node

# EVOLVE-BLOCK-END
