"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math

_DEADLINE = None
_DEADLINE_LIMIT = 145.0  # seconds wall-clock safety margin (evaluator kills at 150s)


def _elapsed():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = _time.monotonic() + _DEADLINE_LIMIT
    return _time.monotonic() - _DEADLINE


def _remaining_time():
    return _DEADLINE_LIMIT - _elapsed()


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Strategy: Multi-criteria dispatch rule combining:
    1. Critical path / remaining work (higher is better)
    2. Machine load balancing (prefer less-loaded machines)
    3. Earliest completion time on best machine
    4. Slight randomness for diversity to avoid local optima
    
    Returns index of the operation to start. Optionally returns (index, machine).
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Compute scores for each ready operation
    # We want to maximize a composite score
    
    best_score = -1e18
    best_idx = 0
    best_machine = None
    
    # Pre-compute some statistics
    total_remaining_work = 0
    max_remaining_work = 0
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        total_remaining_work += rw
        if rw > max_remaining_work:
            max_remaining_work = rw
    
    # Normalize factor
    norm_work = max_remaining_work if max_remaining_work > 0 else 1.0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            # No machine options, skip or give low score
            score = 0
            best_machine_for_op = None
        else:
            # Find the best machine for this operation
            # Criteria for machine selection:
            # - Shorter duration is better
            # - Machine with fewer remaining operations is better (load balancing)
            
            best_machine_for_op = None
            best_machine_score = -1e18
            
            for machine, duration in machine_options.items():
                # Machine score: prefer shorter duration
                # Also consider if we can infer machine load from the environment
                # We don't have direct access to machine state, so use duration as proxy
                m_score = -duration  # shorter is better (negative)
                
                # Add a small bonus for machine diversity (hash-based deterministic tiebreak)
                # This helps avoid always picking the same machine when durations are equal
                machine_hash = hash(machine) % 1000
                m_score += machine_hash * 0.001  # tiny tiebreaker
                
                if m_score > best_machine_score:
                    best_machine_score = m_score
                    best_machine_for_op = machine
            
            # The duration on the best machine
            best_duration = machine_options.get(best_machine_for_op, 0) if best_machine_for_op else 0
            
            # Composite score for this operation:
            # 1. Remaining work (normalized) - higher is better
            work_component = remaining_work / norm_work
            
            # 2. Urgency: fewer remaining ops means this job is closer to completion
            #    Completing jobs early can free up resources
            urgency_component = 1.0 / (1.0 + remaining_ops) if remaining_ops > 0 else 1.0
            
            # 3. Processing time on best machine - shorter is better for makespan
            #    But we also want to keep machines busy, so balance
            time_component = 1.0 / (1.0 + best_duration) if best_duration > 0 else 1.0
            
            # 4. Critical path estimate: remaining_work / max parallelism
            #    Approximate by remaining_work / (1 + remaining_ops)
            critical_component = work_component / (1.0 + remaining_ops * 0.1)
            
            # Weighted combination
            score = (
                0.4 * work_component +
                0.3 * urgency_component +
                0.15 * time_component +
                0.15 * critical_component
            )
            
            # Add a small deterministic variation based on job/op to break ties
            job = op.get("job", 0)
            op_num = op.get("op", 0)
            tie_break = ((job * 7 + op_num * 13) % 100) * 0.001
            score += tie_break
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
    
    # Return just the index (the interface allows (index, machine) but index alone is safe)
    return int(best_idx)
# EVOLVE-BLOCK-END
