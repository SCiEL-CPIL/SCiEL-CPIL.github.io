"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill


def _within_deadline():
    return _time.time() < _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Greedy approximation of LPT for makespan minimization.
    We select the ready operation that maximizes a composite score combining:
      1. The minimum processing time across machine options (inflexibility/longest critical path)
      2. The total remaining work (to prioritize jobs with more work left)
      3. The number of remaining ops (jobs with fewer remaining ops are more urgent)
    
    This balances machine loads early and prevents late-stage bottlenecks.
    """
    if not ready_operations:
        return 0
    
    if not _within_deadline():
        # Safety: return first available
        return 0
    
    n = len(ready_operations)
    
    # For small n, compute a rich score; for large n, use lighter heuristics
    if n <= 200:
        scores = []
        for op in ready_operations:
            # Get machine options
            mopts = op.get("machine_options", {})
            if not mopts:
                scores.append(0.0)
                continue
            
            durations = list(mopts.values())
            min_dur = min(durations)
            max_dur = max(durations)
            mean_dur = sum(durations) / len(durations)
            
            # Remaining work and ops
            rem_work = op.get("remaining_work", 0)
            rem_ops = op.get("remaining_ops", 1)
            
            # Composite score:
            # - Primary: min_duration (LPT-like: prioritize long inflexible tasks)
            # - Secondary: remaining_work (more work = more urgent)
            # - Tertiary: fewer remaining ops = more urgent (closer to completion)
            # - Penalty for high variance in durations (flexible jobs can wait)
            
            # Normalize-ish: use raw values but weight appropriately
            score = 0.0
            score += 2.0 * min_dur          # Strong preference for long min-duration
            score += 0.5 * rem_work          # Moderate preference for more remaining work
            score += 1.0 / max(rem_ops, 1)  # Slight preference for fewer remaining ops
            score -= 0.1 * (max_dur - min_dur)  # Slight penalty for high flexibility
            
            scores.append(score)
        
        # Find argmax
        best_idx = 0
        best_score = scores[0]
        for i in range(1, n):
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        
        return int(best_idx)
    else:
        # Large n: use simpler heuristic to stay fast
        best_idx = 0
        best_min_dur = -1
        for i, op in enumerate(ready_operations):
            mopts = op.get("machine_options", {})
            if not mopts:
                continue
            min_dur = min(mopts.values())
            if min_dur > best_min_dur:
                best_min_dur = min_dur
                best_idx = i
        return int(best_idx)
# EVOLVE-BLOCK-END
