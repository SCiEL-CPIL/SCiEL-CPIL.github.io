<<<<<<< SEARCH
def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1)
    such that every t-subset is contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Phase 1: Greedy set cover with tracking
    blocks, triple_to_blocks, block_triple_indices = _greedy_cover_with_tracking(v, k, t, deadline)
    
    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant(blocks, v, k, t, deadline)
    
    # Phase 3: Local search for further reduction
    blocks = _local_search(blocks, v, k, t, deadline)
    
    return blocks
=======
def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1)
    such that every t-subset is contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Run multiple randomized greedy instances and keep the best
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds for diverse solutions
    num_trials = 5
    time_per_trial = (deadline - start_time) / num_trials
    
    for trial in range(num_trials):
        trial_start = time.time()
        trial_deadline = trial_start + time_per_trial
        seed = 42 + trial * 1000
        
        # Phase 1: Greedy set cover with tracking (randomized)
        blocks, triple_to_blocks, block_triple_indices = _greedy_cover_with_tracking(v, k, t, trial_deadline, seed=seed)
        
        # Phase 2: Remove redundant blocks
        blocks = _remove_redundant(blocks, v, k, t, trial_deadline)
        
        # Phase 3: Local search for further reduction
        blocks = _local_search(blocks, v, k, t, trial_deadline, seed=seed)
        
        # Track best
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Check if we're running out of time
        if time.time() > deadline - 10.0:
            break
    
    # Final polish: one more local search pass on the best solution
    if best_blocks is not None and time.time() < deadline - 5.0:
        best_blocks = _local_search(best_blocks, v, k, t, deadline, seed=9999)
    
    if best_blocks is None:
        # Fallback: single greedy run
        blocks, _, _ = _greedy_cover_with_tracking(v, k, t, deadline, seed=42)
        best_blocks = blocks
    
    return best_blocks
>>>>>>> REPLACE
