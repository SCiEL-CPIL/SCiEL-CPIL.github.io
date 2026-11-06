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
    crit_slack = np.zeros(n, dtype=float)  # smooth critical-path slack
    
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
        
        # Soft-min via log-sum-exp (smooth surrogate for min)
        min_d = min_durs[i]
        exp_terms = np.exp(-(durations - min_d))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_durs[i] = min_d - log_sum
        
        if soft_min_durs[i] <= 0:
            soft_min_durs[i] = 0.1
        
        # Smooth critical-path slack: remaining_work / (remaining_ops * soft_min_dur)
        # Higher = more critical (less slack per remaining op)
        rop = remaining_ops_arr[i]
        sm = soft_min_durs[i]
        if rop * sm > 0:
            crit_slack[i] = remaining_works[i] / (rop * sm)
        else:
            crit_slack[i] = 0.0
        
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
    
    for i in range(n):
        if not valid[i]:
            continue
        
        rw = remaining_works[i]
        mn = min_durs[i]
        sm = soft_min_durs[i]
        no = num_opts[i]
        rop = remaining_ops_arr[i]
        
        # Feature 1: Constraint tightness (inverse of number of options)
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
        
        # Feature 9: Smooth critical-path weight (using soft-min)
        f9 = crit_slack[i]
        
        # Weighted combination
        w1 = 2.5   # constraint tightness
        w2 = 1.0   # remaining work
        w3 = 2.0   # SPT (soft)
        w4 = 0.5   # remaining ops
        w5 = 4.5   # WSPT - strong signal
        w6 = 1.0   # contention penalty
        w7 = 0.5   # machine clarity
        w8 = 1.5   # machine load penalty
        w9 = 3.5   # smooth critical path weight
        
        score = (w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 
                 + w5 * f5 - w6 * f6 + w7 * f7 - w8 * f8 + w9 * f9)
        
        scores[i] = score
    
    # Pick the operation with the highest score, with a polish tie-break
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return 0
    
    # Primary selection
    primary_scores = scores[valid_indices]
    best_local = np.argmax(primary_scores)
    best_idx = valid_indices[best_local]
    best_score = primary_scores[best_local]
    
    # Polish pass: if top candidates are very close, use a secondary scoring
    # that emphasizes different aspects to break ties more informatively
    close_threshold = 0.15
    close_mask = primary_scores >= (best_score - close_threshold)
    close_indices = valid_indices[close_mask]
    
    if len(close_indices) > 1:
        # Secondary score: emphasize constraint tightness and critical path more
        sec_scores = np.zeros(len(close_indices), dtype=float)
        for j, ci in enumerate(close_indices):
            rw = remaining_works[ci]
            no = num_opts[ci]
            cs = crit_slack[ci]
            mn = min_durs[ci]
            sm = soft_min_durs[ci]
            # Secondary: heavy weight on tightness + critical path, light on rest
            sec_scores[j] = (4.0 / no + 5.0 * cs + 1.0 * (1.0 / sm) - 0.5 * (mn / total_work))
        
        # Pick the best secondary score among close candidates
        best_sec_local = np.argmax(sec_scores)
        best_idx = close_indices[best_sec_local]
    
    # Return (index, machine) to let simulator pick the best machine
    bm = best_machines[best_idx]
    if bm is not None:
        return (int(best_idx), bm)
    
    return int(best_idx)
# EVOLVE-BLOCK-END
