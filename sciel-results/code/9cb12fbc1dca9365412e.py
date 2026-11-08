"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Global start time for deadline guard
_START_TIME = _time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    Uses a heuristic based on Remaining Work / Minimum Processing Time.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we are running low on time, fall back to a very fast heuristic
    # Although this function is fast, we check just in case of unexpected overhead
    if _time.time() > _DEADLINE:
        # Fallback: First ready operation
        return 0

    best_idx = 0
    best_score = -1.0

    for i, op in enumerate(ready_operations):
        # Find the minimum processing time among eligible machines
        min_time = min(op["machine_options"].values())
        
        # Avoid division by zero
        if min_time < 1e-9:
            min_time = 1e-9
            
        # Heuristic: Prioritize operations with high remaining work and low processing time
        # This is similar to WSPT (Weighted Shortest Processing Time)
        score = op["remaining_work"] / min_time
        
        # Tie-breaker: If scores are very close, prefer operations with more remaining ops
        # This helps in keeping jobs in flow
        if score > best_score + 1e-9:
            best_score = score
            best_idx = i
        elif abs(score - best_score) <= 1e-9:
            # Tie-break: More remaining operations is often better for makespan
            if op["remaining_ops"] > ready_operations[best_idx]["remaining_ops"]:
                best_idx = i

    return best_idx
# EVOLVE-BLOCK-END
