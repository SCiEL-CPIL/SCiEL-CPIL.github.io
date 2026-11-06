"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import math

# Deadline guard
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0

def _elapsed():
    return _time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        # Fast fallback: pick max remaining_work
        best_idx = 0
        best_rw = -1.0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        return int(best_idx)
    
    n = len(ready_operations)
    
    # Track machine loads from ready operations to model contention
    machine_load = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo.keys():
            machine_load[m] = machine_load.get(m, 0.0) + 1.0
    
    # Pre-extract features
    remaining_works = np.zeros(n, dtype=float)
    min_durs = np.ones(n, dtype=float)
    soft_min_durs = np.ones(n, dtype=float)
    num_opts = np.ones(n, dtype=float)
    remaining_ops_arr = np.ones(n, dtype=float)
    valid = np.ones(n, dtype=bool)
    best_machines = [None] * n
    min_dur_per_op = [None] * n
    
    for i, op in enumerate(ready_operations):
        remaining_works[i] = op.get("remaining_work", 0.0)
        machine_options = op.get("machine_options", {})
        remaining_ops_arr[i] = max(op.get("remaining_ops", 0), 1)
        
        if not machine_options:
            valid[i] = False
            continue
        
        durations = np.array(list(machine_options.values()), dtype=float)
        num_opts[i] = len(durations)
        
        # Clamp to positive
        durations = np.clip(durations, 0.1, None)
        
        min_durs[i] = np.min(durations)
        
        # Soft-min via log-sum-exp
        min_d = min_durs[i]
        exp_terms = np.exp(-(durations - min_d))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_durs[i] = min_d - log_sum
        
        if soft_min_durs[i] <= 0:
            soft_min_durs[i] = 0.1
        
        # Find the best machine (shortest duration, tie-break by lower load)
        best_m = None
        best_score_m = float('inf')
        for m, d in machine_options.items():
            load_penalty = machine_load.get(m, 0.0) * 0.1
            m_score = d + load_penalty
            if m_score < best_score_m:
                best_score_m = m_score
                best_m = m
        best_machines[i] = best_m
        min_dur_per_op[i] = min_durs[i]
    
    # Compute scores
    scores = np.full(n, float('-inf'), dtype=float)
    total_work = np.sum(remaining_works[valid])
    if total_work <= 0:
        total_work = 1.0
    
    # Compute average duration per machine for idle-time projection
    machine_total_dur = {}
    machine_count = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m, d in mo.items():
            machine_total_dur[m] = machine_total_dur.get(m, 0.0) + max(d, 0.1)
            machine_count[m] = machine_count.get(m, 0) + 1
    machine_avg_dur = {}
    for m in machine_total_dur:
        machine_avg_dur[m] = machine_total_dur[m] / max(machine_count[m], 1)
    
    for i in range(n):
        if not valid[i]:
            continue
        
        rw = remaining_works[i]
        mn = min_durs[i]
        sm = soft_min_durs[i]
        no = num_opts[i]
        rop = remaining_ops_arr[i]
        
        # Feature 1: Constraint tightness (inverse of number of options)
        # Strongly weight operations with few machine options to prevent starvation
        f1 = 1.0 / no
        
        # Feature 2: Remaining work urgency
        f2 = rw
        
        # Feature 3: Inverse soft-min duration (SPT-like)
        f3 = 1.0 / sm
        
        # Feature 4: Inverse remaining ops (critical path proxy)
        f4 = 1.0 / rop
        
        # Feature 5: WSPT proxy = work / duration
        f5 = rw / sm
        
        # Feature 6: Contention penalty
        f6 = mn / total_work
        
        # Feature 7: Machine clarity ratio
        f7 = mn / sm if sm > 0 else 1.0
        
        # Feature 8: Machine load penalty for the chosen machine
        bm = best_machines[i]
        ml = machine_load.get(bm, 0.0) if bm is not None else 0.0
        f8 = ml
        
        # Feature 9: Critical path weight = remaining work / (min_dur * remaining_ops)
        # Higher means this op is more critical to the overall makespan
        f9 = rw / (mn * rop) if mn * rop > 0 else 0.0
        
        # Feature 10: Machine idle projection
        # If the chosen machine has low queue relative to this op's duration,
        # starting this op now fills idle time (good). Penalize ops that would
        # leave the machine more idle relative to its average workload.
        if bm is not None and mn > 0:
            avg_d = machine_avg_dur.get(bm, mn)
            # Projected idle ratio: how much the machine's average job is longer
            # than this op. If this op is much shorter than avg, it might not
            # fully utilize the machine slot.
            idle_ratio = max(0.0, (avg_d - mn) / avg_d) if avg_d > 0 else 0.0
            f10 = idle_ratio
        else:
            f10 = 0.0
        
        # Feature 11: Constraint urgency — ops with very few options AND high
        # remaining work are most critical
        f11 = (1.0 / no) * (rw / total_work) if total_work > 0 else 0.0
        
        # Weighted combination
        w1 = 3.5   # constraint tightness - strengthened
        w2 = 0.8   # remaining work
        w3 = 2.0   # SPT (soft)
        w4 = 0.5   # remaining ops
        w5 = 4.0   # WSPT - reduced slightly
        w6 = 1.0   # contention penalty
        w7 = 0.5   # machine clarity
        w8 = 1.5   # machine load penalty
        w9 = 3.0   # critical path weight
        w10 = 1.0  # machine idle projection penalty
        w11 = 2.0  # constraint urgency
        
        score = (w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 
                 + w5 * f5 - w6 * f6 + w7 * f7 - w8 * f8 + w9 * f9
                 - w10 * f10 + w11 * f11)
        
        scores[i] = score
    
    # Pick the operation with the highest score
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return 0
    
    best_idx = valid_indices[np.argmax(scores[valid_indices])]
    
    # Return (index, machine) to let simulator pick the best machine
    bm = best_machines[best_idx]
    if bm is not None:
        return (int(best_idx), bm)
    
    return int(best_idx)
# EVOLVE-BLOCK-END
