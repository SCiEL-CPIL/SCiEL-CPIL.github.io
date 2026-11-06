"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Compute priority scores for vertices using a dynamic simulation approach.
    We simulate the greedy decoder process multiple times with different random
    perturbations to the base priority (inverse degree). By observing which vertices
    get selected early across these simulations, we identify robust 'good' vertices
    that are likely to be part of a large independent set.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Base score: inverse degree (low degree is good for independent set)
    degrees = adj.sum(axis=1)
    # Avoid division by zero
    safe_degrees = np.where(degrees > 0, degrees, 1.0)
    base_scores = 1.0 / safe_degrees
    
    # We will run multiple "virtual" greedy decoders in parallel.
    # Each decoder has a slightly different random noise added to base_scores.
    # This allows us to explore the space of possible orderings.
    
    # Determine number of simulations based on time budget
    # Each simulation step involves matrix operations.
    # Let's start with a moderate number, say 50-100, and see how fast it is.
    # n=200, adj is 200x200. matmul is fast.
    
    n_sims = 100
    noise_scale = 0.1
    
    # Generate random noise for each simulation
    # Shape: (n_sims, n)
    noise = np.random.randn(n_sims, n) * noise_scale
    
    # Initial scores for all simulations
    # scores_sim[i, j] is the score of vertex j in simulation i
    scores_sim = base_scores[None, :] + noise
    
    # Track which vertices are "active" (not yet skipped) in each simulation
    # active_sim[i, j] is True if vertex j is still a candidate in simulation i
    active_sim = np.ones((n_sims, n), dtype=bool)
    
    # Track how many times each vertex is selected as the "top" active vertex
    # selection_count[j] counts how often vertex j is picked
    selection_count = np.zeros(n, dtype=float)
    
    # We simulate the greedy process for a fixed number of steps or until all are picked.
    # Since we want a priority score, we want to know the *order* or *importance*.
    # A vertex that is picked early in many simulations is likely a good seed.
    # Alternatively, we can just count how many times it is selected.
    # But the greedy decoder picks in descending score.
    # If we simulate the full greedy process, the first vertex picked in each sim is the one with max score.
    # Then it is removed, and its neighbors are removed.
    # We can accumulate a score based on the "step" at which it is picked.
    # Earlier step -> higher score.
    
    # Let's accumulate a weighted score: sum of (max_step - step_picked)
    # Or simply: if picked at step k, add (n - k) to its score.
    
    max_steps = n
    # Pre-allocate for speed?
    # We need to find the max active score in each sim, pick it, and mask it + neighbors.
    
    # To speed up, we can vectorize the "pick max" part across simulations.
    # However, the masking depends on which vertex was picked, which differs per sim.
    # This makes full vectorization of the loop tricky.
    # But n=200 is small. n_sims=100. 100*200 = 20,000 iterations.
    # Each iteration:
    # 1. Mask inactive scores with -inf
    # 2. Find max index per row
    # 3. Update active_sim for those rows
    # 4. Update selection_count
    
    # Let's try to be as efficient as possible.
    
    # We can use np.where to mask.
    # For each step:
    # masked_scores = np.where(active_sim, scores_sim, -np.inf)
    # max_idx = np.argmax(masked_scores, axis=1)
    # Now, for each sim i, we picked vertex v_i = max_idx[i].
    # We need to set active_sim[i, v_i] = False and active_sim[i, neighbors(v_i)] = False.
    
    # This loop is over steps (up to n).
    # Inside, we operate on arrays of size n_sims.
    
    elapsed_check_interval = 10
    step = 0
    
    while step < max_steps:
        # 1. Mask inactive
        masked_scores = np.where(active_sim, scores_sim, -np.inf)
        
        # 2. Find max index
        max_idx = np.argmax(masked_scores, axis=1)
        
        # 3. Check if any sim is still active
        # If a row in active_sim is all False, argmax will return 0 (arbitrary).
        # We should only update active sims.
        # Let's find which sims are still active
        active_sims_idx = np.where(np.any(active_sim, axis=1))[0]
        
        if len(active_sims_idx) == 0:
            break
            
        # For these active sims, we have a valid max_idx
        # We need to update active_sim for these specific sims.
        
        # Vectorized update is hard because neighbors differ per picked vertex.
        # We can loop over the active sims? No, that's n_sims * n steps = 20,000 * 100 = 2M ops?
        # Wait, the loop is over steps (200). Inside, we process all active sims (up to 100).
        # If we loop over sims inside the step loop, that's 200 * 100 = 20,000 iterations.
        # Each iteration does small array ops. This is fast in Python?
        # 20,000 iterations is very fast.
        
        # Let's try a hybrid: vectorize the "pick" but loop for the "mask neighbors" if necessary.
        # Actually, we can just loop over the active simulations.
        
        for i in active_sims_idx:
            v = max_idx[i]
            
            # Add to score: earlier picks are better.
            # Score contribution: (max_steps - step)
            selection_count[v] += (max_steps - step)
            
            # Deactivate v
            active_sim[i, v] = False
            
            # Deactivate neighbors of v
            # Get neighbors of v
            neighbors = np.where(adj[v] > 0)[0]
            # Only deactivate if they are currently active
            active_sim[i, neighbors] = False
        
        step += 1
        
        # Time check
        if step % elapsed_check_interval == 0:
            elapsed = time.time() - start_time
            if elapsed > 100:
                break
        
        if step >= max_steps:
            break

    # The selection_count now reflects how "important" each vertex is.
    # Vertices picked early get higher counts.
    # We can return selection_count as the priority.
    # But we should normalize or scale it?
    # The greedy decoder sorts by score. Higher score -> picked first.
    # So we want vertices with high selection_count to have high priority.
    
    # To ensure stability and distinctness, we can add a small random noise or use the base score as tie-breaker.
    # But the task says "fixed greedy decoder picks vertices in descending score order".
    # So if we return selection_count, the decoder will pick the vertex with the highest count first.
    
    # One issue: if selection_count is all zeros (e.g., if we broke too early), we fall back to base_scores.
    # Also, we might want to combine with base_scores to handle cases where simulation is noisy.
    
    # Let's blend: 0.9 * normalized_selection_count + 0.1 * base_scores
    # Or just use selection_count.
    
    # Let's check if we completed the simulation.
    # If not, selection_count might be biased towards vertices picked in the first few steps.
    # That's actually fine, as early picks are more critical.
    
    # To make it a proper priority, we can just return selection_count.
    # But we should ensure it's a float array.
    
    final_scores = selection_count.astype(float)
    
    # If all scores are 0 (should not happen if n>0 and we ran at least 1 step), fallback
    if np.all(final_scores == 0):
        final_scores = base_scores
        
    return final_scores

# EVOLVE-BLOCK-END
