"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling using a Lexicographic Heuristic.
    
    Approach:
    1. Prioritize operations with fewer machine options (less flexibility) to resolve
       constrained tasks early.
    2. Among those, prioritize operations with the most remaining work (MW) to
       address potential bottlenecks.
    3. Break ties by shortest minimum duration to keep throughput high.
    
    Returns the index of the best operation to start next.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Calculate features for lexicographic sorting
    # We want to pick the op that is "most critical".
    # Criteria:
    # 1. Fewest machine options (tightest constraint)
    # 2. Most remaining work (largest bottleneck potential)
    # 3. Shortest min duration (fastest execution on best machine)
    
    best_idx = 0
    best_key = None
    
    for i, op in enumerate(ready_operations):
        # Feature 1: Number of machine options (lower is better)
        num_options = len(op["machine_options"])
        
        # Feature 2: Remaining work (higher is better)
        remaining_work = op["remaining_work"]
        
        # Feature 3: Min duration (lower is better)
        if op["machine_options"]:
            min_dur = min(op["machine_options"].values())
        else:
            min_dur = 0
            
        # Create a tuple key for comparison
        # We want:
        # - num_options ASC (smaller is better)
        # - remaining_work DESC (larger is better) -> negate it
        # - min_dur ASC (smaller is better)
        
        key = (num_options, -remaining_work, min_dur)
        
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
