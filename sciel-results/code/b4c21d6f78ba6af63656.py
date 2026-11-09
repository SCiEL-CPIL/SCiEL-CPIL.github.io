"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math
import heapq

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: MWR
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    try:
        if not ready_operations:
            return 0
        
        n = len(ready_operations)
        
        # 1. Calculate features for all ready operations
        # We want to estimate the "criticality" and "efficiency" of each op.
        
        # Feature: Remaining Work (MWR) - Higher is generally better for makespan reduction
        # Feature: Min Duration (SPT) - Lower is better for machine utilization
        # Feature: Number of Machines - Higher flexibility might be better? Or lower constraint?
        
        # Let's use a composite score.
        # Common heuristics for FJSP:
        # - MWR (Most Remaining Work)
        # - SPT (Shortest Processing Time)
        # - CR (Critical Ratio) - not applicable without due dates
        # - Slack - not applicable
        
        # A robust approach is often a weighted combination.
        # Let's try: Score = w1 * (remaining_work / max_remaining_work) + w2 * (min_duration / max_min_duration)
        # Wait, for min_duration, we want to MINIMIZE it, so we should invert it or subtract.
        # Let's maximize:
        # Utility = alpha * (remaining_work / max_rw) + beta * (1 / (min_d + epsilon))
        
        # Or, simpler:
        # Prioritize ops with high remaining work.
        # Among those, pick the one with the shortest processing time on its best machine.
        
        max_rw = max(op.get("remaining_work", 0) for op in ready_operations)
        if max_rw == 0:
            max_rw = 1.0
            
        best_idx = 0
        best_score = -1.0
        
        # Pre-compute min duration for each op to avoid repeated min() calls in the loop if we do complex scoring
        # But for N=10, it's fast enough.
        
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            mo = op.get("machine_options", {})
            
            if not mo:
                min_d = 1.0
            else:
                min_d = min(mo.values())
            
            # Score components
            # 1. Work Urgency: How much work is left?
            #    Normalize to [0, 1]
            work_score = rw / max_rw
            
            # 2. Efficiency: How fast can it be done?
            #    We want small min_d to be high score.
            #    Use 1/(min_d + 1) or similar.
            #    Let's check the scale. Durations are usually small integers.
            eff_score = 1.0 / (min_d + 1.0)
            
            # 3. Flexibility: Number of machine options
            #    More options might allow better load balancing, but for immediate dispatch,
            #    it doesn't change the current op's duration (we pick min).
            #    However, if we have to return (index, machine), we pick the machine with min_d.
            #    For now, let's just return index. The problem says "optionally (index, machine)".
            #    The current program returns index. Let's stick to index to be safe, unless
            #    the evaluator strictly expects a tuple?
            #    "dispatch(ready_operations, time) -> index of the ready operation to start (optionally (index, machine))"
            #    This implies returning just the index is valid.
            
            # Composite Score
            # Weights: Work is usually dominant for makespan.
            # Let's try 0.7 * work + 0.3 * eff
            score = 0.7 * work_score + 0.3 * eff_score
            
            # Tie breaker: If scores are very close, prefer lower min_d?
            # Or prefer higher rw?
            
            if score > best_score:
                best_score = score
                best_idx = i
            elif abs(score - best_score) < 1e-9:
                # Tie-break: prefer shorter min_d
                if min_d < min(ready_operations[best_idx].get("machine_options", {0:1}).values()):
                    best_idx = i
        
        return int(best_idx)

    except Exception:
        # Robust fallback
        if not ready_operations:
            return 0
        return 0

# EVOLVE-BLOCK-END
