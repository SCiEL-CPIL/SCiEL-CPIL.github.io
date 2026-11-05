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
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # Initialize scores to -1 (unselected)
    scores = np.full(n, -1.0)
    
    # Track which vertices are still available (not blocked)
    available = np.ones(n, dtype=bool)
    
    # Current adjacency matrix of the "active" subgraph
    # We need to know neighbors that are also available.
    # Instead of updating a matrix, we can just check availability when counting neighbors.
    
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
        # We can vectorize this: for each candidate i, sum(available & adjacency[i, :])
        
        # Efficient way:
        # active_degrees = np.zeros(n)
        # For all i in candidates:
        #    active_degrees[i] = np.sum(available & adjacency[i, :])
        
        # Using matrix multiplication or logical operations:
        # active_degrees = (adjacency.astype(int) @ available.astype(int))
        # This computes the number of available neighbors for ALL vertices.
        # We only care about candidates.
        
        active_neighbor_counts = adjacency.astype(float) @ available.astype(float)
        
        # Get active degrees for candidates
        cand_active_degrees = active_neighbor_counts[candidates]
        
        # Select candidates with the minimum active degree
        min_degree = np.min(cand_active_degrees)
        
        # Find all candidates that have this minimum degree
        # We want to select ALL of them in this round (parallel selection)
        # This corresponds to the "parallel" aspect of the exclusion process.
        
        # Ties are broken by index to ensure determinism, but since we select all with min degree,
        # we just need to identify them.
        
        # Create a mask for candidates to select
        # We want indices in `candidates` where `cand_active_degrees` == `min_degree`
        
        # To avoid floating point issues, use a small tolerance or exact match if integers.
        # Degrees are integers, so float representation is exact for small N.
        
        select_mask_in_candidates = (cand_active_degrees == min_degree)
        
        # Get the actual indices to select
        to_select = candidates[select_mask_in_candidates]
        
        if len(to_select) == 0:
            # Should not happen
            break
            
        # Assign score: higher score for earlier round
        # Score = max_rounds - current_round (so round 0 gets highest score)
        # Or just -current_round. Since we want descending score for greedy decoder,
        # earlier rounds should have HIGHER scores.
        scores[to_select] = max_rounds - current_round
        
        # Update available status
        # Vertices selected are no longer available.
        # Their neighbors are also no longer available (blocked).
        
        available[to_select] = False
        
        # Block neighbors
        # neighbors_of_selected = np.where(adjacency[to_select, :].any(axis=0))[0]
        # available[neighbors_of_selected] = False
        
        # More efficient:
        # Union of neighbors of all selected vertices
        # We can do this by iterating or using matrix ops.
        # Since N=200, even O(N^2) is fine.
        
        # Let's use a boolean mask for neighbors
        # adj_to_selected = adjacency[:, to_select] # N x K matrix
        # any_neighbor_selected = adj_to_selected.any(axis=1)
        # available[any_neighbor_selected] = False
        
        # This blocks the selected vertices themselves too (already done) and their neighbors.
        # We must ensure we don't un-block anyone.
        
        # Note: If a vertex was already blocked, it stays blocked.
        # available is only ever set to False.
        
        # Compute neighbors of selected
        # adjacency is symmetric? Yes, undirected graph.
        # neighbors = np.zeros(n, dtype=bool)
        # for idx in to_select:
        #     neighbors |= adjacency[idx, :]
        
        # Vectorized:
        # neighbors = adjacency[:, to_select].any(axis=1)
        
        available[adjacency[:, to_select].any(axis=1)] = False
        
        current_round += 1
        
    # If deadline exceeded or loop finished, ensure unassigned vertices get a low score
    # They were never selected in the simulation, so they are "blocked" by previous selections
    # or the process didn't reach them.
    # Assign them a score lower than any selected round.
    # min_score = max_rounds - max_rounds = 0
    # So unassigned get -1 or similar.
    # Actually, if a vertex is available at the end but not selected (shouldn't happen if loop runs to completion),
    # it means it was blocked? No, if it's available, it would be selected in the next round.
    # So if the loop terminates due to deadline, some might be unassigned.
    
    # Assign 0 to unassigned (score -1 was init, let's make it lower)
    # Actually, the greedy decoder picks highest score first.
    # If we assign -1 to unassigned, they are picked last.
    # But wait, if they are unassigned, it means the simulation didn't pick them.
    # This implies they are neighbors of picked vertices?
    # In the simulation, we mark neighbors as unavailable.
    # So any vertex that remains "available" at the end of a full run (all rounds done)
    # would have been selected.
    # If we stop early due to deadline, some are unassigned.
    # We should give them a very low priority.
    
    scores[scores < 0] = -1.0
    
    # Add a tiny tie-breaker based on index to ensure strict ordering if needed?
    # The greedy decoder might handle ties arbitrarily.
    # Adding a small epsilon based on index can help stability.
    # scores = scores - np.arange(n) * 1e-6
    
    return scores
# EVOLVE-BLOCK-END
