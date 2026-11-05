"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds


def _is_deadline_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def priority(adjacency):
    """
    Compute priority scores based on a parallel discrete-time exclusion process.
    
    We simulate a process where in each round, all vertices that are not yet "blocked"
    (i.e., not neighbors of previously selected vertices) are considered.
    Among these candidates, we select those with the fewest active neighbors.
    This mimics a greedy strategy that prioritizes vertices that block the fewest
    other potential candidates, effectively peeling the graph.
    
    The priority score is determined by the round in which the vertex is selected.
    Vertices selected in earlier rounds get higher priority scores.
    
    Improvements:
    - Use integer matrix for faster multiplication
    - Add tie-breaking by original degree (prefer lower degree in ties)
    - Add small epsilon based on index for stable ordering
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # Convert to int8 for faster matrix-vector multiplication
    adj_int = adjacency.astype(np.int8)
    
    # Compute original degrees for tie-breaking
    original_degrees = adjacency.sum(axis=1).astype(np.float64)
    
    # Initialize scores to -1 (unselected)
    scores = np.full(n, -1.0)
    
    # Track which vertices are still available (not blocked)
    available = np.ones(n, dtype=bool)
    
    current_round = 0
    max_rounds = n  # Safety limit
    
    while available.sum() > 0 and current_round < max_rounds:
        if _is_deadline_exceeded():
            break
            
        # Find candidates: vertices that are available
        candidates = np.where(available)[0]
        
        if len(candidates) == 0:
            break
            
        # For each candidate, count how many of its neighbors are also available.
        # This is the "active degree".
        # Using matrix multiplication: adj_int @ available_int gives count of available neighbors for all vertices
        active_neighbor_counts = adj_int @ available.astype(np.int8)
        
        # Get active degrees for candidates
        cand_active_degrees = active_neighbor_counts[candidates].astype(np.float64)
        
        # Select candidates with the minimum active degree
        min_degree = np.min(cand_active_degrees)
        
        # Find all candidates that have this minimum degree
        select_mask_in_candidates = (cand_active_degrees == min_degree)
        
        # Get the actual indices to select
        to_select = candidates[select_mask_in_candidates]
        
        if len(to_select) == 0:
            # Should not happen
            break
            
        # Assign score: higher score for earlier round
        # Score = max_rounds - current_round (so round 0 gets highest score)
        # Add a small tie-breaker: prefer vertices with lower original degree
        # This encourages selecting peripheral vertices first
        base_score = max_rounds - current_round
        # Subtract a small amount proportional to original degree to break ties
        # Lower degree -> higher effective score
        degree_penalty = original_degrees[to_select] * 1e-4
        scores[to_select] = base_score - degree_penalty
        
        # Update available status
        # Vertices selected are no longer available.
        # Their neighbors are also no longer available (blocked).
        
        available[to_select] = False
        
        # Block neighbors using boolean matrix operations
        # adjacency[:, to_select] gives N x K boolean matrix
        # .any(axis=1) gives N boolean vector: True if vertex is neighbor of any selected
        available[adjacency[:, to_select].any(axis=1)] = False
        
        current_round += 1
        
    # If deadline exceeded or loop finished, ensure unassigned vertices get a low score
    # They were never selected in the simulation
    scores[scores < 0] = -1.0
    
    # Add a tiny tie-breaker based on index to ensure strict ordering
    # This helps with stability in the greedy decoder
    scores = scores - np.arange(n) * 1e-8
    
    return scores
# EVOLVE-BLOCK-END
