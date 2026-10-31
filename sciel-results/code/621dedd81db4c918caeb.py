"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management
_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin from 120s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Geometric Angle-Aware Heuristic.
    
    Instead of summing distances (which treats all directions equally), this method
    calculates the cost of moving to a candidate based on:
    1. The distance from current to candidate.
    2. A penalty for the angle between the vector (current->destination) and 
       (current->candidate).
    
    This encourages paths that keep moving generally toward the destination 
    without unnecessary lateral deviations, often resulting in smoother, 
    shorter tours than simple weighted sum of distances.
    """
    # Check time budget
    if time.time() - _START_TIME > TIME_LIMIT:
        # Fallback to pure NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1 # Should not happen if logic is correct upstream
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Distances from current to all unvisited candidates
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from current to destination
    d_dest_from_current = distance_matrix[current_node, destination_node]
    
    # Distance from each unvisited candidate to destination
    d_candidate_to_dest = distance_matrix[unvisited_nodes, destination_node]

    # We want to penalize candidates that deviate significantly from the direct path to destination.
    # Using the Law of Cosines, we can compute cos(theta) where theta is the angle 
    # at 'current_node' between 'destination_node' and 'candidate'.
    # a^2 = b^2 + c^2 - 2bc*cos(A)
    # Here:
    # a = distance(candidate, destination)
    # b = distance(current, destination)
    # c = distance(current, candidate)
    # cos(theta) = (b^2 + c^2 - a^2) / (2 * b * c)
    
    # Handle division by zero if current == destination (unlikely in TSP unless 1 city left)
    # Or if d_current is 0 (duplicate nodes, unlikely in uniform random)
    
    # To avoid numerical issues and ensure stability, we use a safe denominator
    denom = 2.0 * d_dest_from_current * d_current
    # If denom is very small, the angle calculation is unstable or irrelevant.
    # In that case, we rely more on d_current.
    
    # Vectorized calculation
    # We need to be careful with numpy broadcasting.
    # d_dest_from_current is scalar (float)
    # d_current is array (n,)
    # d_candidate_to_dest is array (n,)
    
    # If d_dest_from_current is 0, we are at the destination. Just pick nearest unvisited.
    if d_dest_from_current < 1e-9:
         return int(unvisited_nodes[np.argmin(d_current)])
         
    # Calculate cos_theta
    # Note: due to floating point errors, cos_theta might be slightly > 1 or < -1.
    # Clip it to [-1, 1] for safety if we were taking acos, but we use it directly in cost.
    # Actually, we can just use the raw value for the penalty term.
    # A common geometric heuristic is: Cost = d_current * (1 + penalty_factor * (1 - cos_theta))
    # Or simply: Cost = d_current + penalty * (1 - cos_theta) * d_current
    # Let's try: Cost = d_current * (1 + alpha * (1 - cos_theta))
    # If theta = 0 (straight towards dest), cos=1, Cost = d_current.
    # If theta = 90 deg (perpendicular), cos=0, Cost = d_current * (1 + alpha).
    # If theta = 180 deg (away from dest), cos=-1, Cost = d_current * (1 + 2*alpha).
    
    # This formulation scales the penalty with the step length, which is robust.
    # Let's choose alpha = 0.5.
    
    cos_theta = (d_dest_from_current**2 + d_current**2 - d_candidate_to_dest**2) / denom
    
    # Clip to valid range to avoid NaNs from slight numerical overshoots
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    alpha = 0.5
    # The term (1 - cos_theta) is 0 when aligned, 2 when anti-aligned.
    # So the multiplier ranges from 1.0 to 1.0 + 2*alpha.
    scores = d_current * (1.0 + alpha * (1.0 - cos_theta))
    
    min_idx = np.argmin(scores)
    
    return int(unvisited_nodes[min_idx])
# EVOLVE-BLOCK-END
