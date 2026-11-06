import numpy as np
import time as time_module
import heapq
import random

# ------------------------------------------------------------------
#  Open-Shop dispatch rule.
#
#  Strategy: a parameterised multi-criterion heuristic with a small
#  beam-search over the top-K candidates when look-ahead data is
#  available.  The weights (w_mwr, w_spt, w_mach, w_urg) are tuned
#  to favour long-remaining-work jobs while breaking ties with short
#  processing times and machine-load balancing.  A deadline guard
#  ensures we never exceed the 150 s wall-clock budget.
# ------------------------------------------------------------------

_DEADLINE = 150.0
_START_TIME = time_module.time()
_TIME_MARGIN = 2.0  # seconds of safety margin before hard limit

# Tuned weights (parameterised)
W_MWR  = 1.0    # max remaining work
W_SPT  = 0.15   # inverse of min duration (speed)
W_MACH = 0.05   # machine load balance (penalise busy machines)
W_URG  = 0.0    # urgency / critical path weight (reserved)
W_MK   = 0.5    # weight for estimated makespan component (lower is better)

_BEAM_WIDTH = 3
_TOP_K = 5
_LSE_TEMP = 5.0  # temperature for log-sum-exp (lower = sharper max approximation)


def _elapsed():
    return time_module.time() - _START_TIME


def _remaining_time():
    return _DEADLINE - _TIME_MARGIN - _elapsed()


def _score_op(op, machine_loads=None, time=0.0, machine_est_finish=None):
    """
    Score a single ready operation.  Higher score = more preferred.
    Uses a smooth log-sum-exp surrogate for the max-completion-time
    (makespan) objective to avoid the nonsmoothness of plain max().
    """
    rw = op.get("remaining_work", 0.0)
    mo = op.get("machine_options", {})
    
    # Min and mean duration
    durs = list(mo.values()) if mo else [0.0]
    min_dur = min(durs) if durs else 0.0
    mean_dur = sum(durs) / len(durs) if durs else 0.0
    
    # Number of machine options (flexibility)
    n_opts = len(mo) if mo else 1
    
    # Machine load penalty: if we track loads, prefer the least-loaded
    # machine among this op's options.  We approximate by counting
    # how many ready ops share each machine.
    mach_penalty = 0.0
    if machine_loads is not None and mo:
        # Load of each candidate machine
        loads = [machine_loads.get(m, 0) for m in mo.keys()]
        min_load = min(loads) if loads else 0
        mach_penalty = min_load  # lower is better, so we subtract
    
    # Remaining ops count (urgency proxy)
    rem_ops = op.get("remaining_ops", [])
    n_rem = len(rem_ops) if isinstance(rem_ops, (list, tuple)) else 0
    
    # Estimate completion time of this op on its best machine.
    # Use machine_est_finish (estimated time when each machine becomes
    # free) if available, otherwise fall back to `time`.
    if machine_est_finish is not None and mo:
        best_finish = min(machine_est_finish.get(m, time) + mo[m] for m in mo)
    else:
        best_finish = time + min_dur
    
    # Smooth surrogate for makespan contribution:
    #   soft_max(completion, remaining_work) ≈ 
    #   temp * log(exp(comp/temp) + exp(rw/temp))
    # This is a differentiable approximation to max(comp, rw).
    # Lower is better, so we subtract it from the score.
    if _LSE_TEMP > 0:
        t = _LSE_TEMP
        # Guard against overflow: shift by max value
        m_val = max(best_finish, rw, 0.0)
        lse = m_val + t * np.log(np.exp((best_finish - m_val) / t) + np.exp((rw - m_val) / t))
    else:
        lse = max(best_finish, rw)
    
    score = (W_MWR * rw
             + W_SPT * (1.0 / (min_dur + 1.0))
             + W_MACH * (-mach_penalty)
             + W_URG * (n_rem / (n_rem + 1.0))
             - W_MK * lse)
    
    return score


def _count_machine_loads(ready_operations):
    """
    Build a dict mapping machine -> count of ready ops that use it.
    This is a rough proxy for machine congestion.
    """
    loads = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo.keys():
            loads[m] = loads.get(m, 0) + 1
    return loads


def _estimate_machine_finish(ready_operations, time):
    """
    Estimate when each machine becomes free, assuming ready ops are
    scheduled in some order.  We use a greedy approximation: for each
    machine, sum the minimum durations of ready ops that can use it,
    starting from `time`.  This gives an upper bound on when each
    machine is busy.
    """
    est = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        # For each machine this op can use, add its duration as a
        # contribution to that machine's estimated busy time.
        # We use the minimum duration as a lower-bound contribution.
        min_dur = min(mo.values())
        for m in mo.keys():
            est[m] = est.get(m, 0.0) + min_dur
    
    # Convert to estimated finish times
    for m in est:
        est[m] = time + est[m]
    
    return est


def _can_look_ahead(ready_operations):
    """
    Check whether the remaining_ops field contains structured data
    that allows a one-step look-ahead simulation.
    """
    if not ready_operations:
        return False
    sample = ready_operations[0]
    rem = sample.get("remaining_ops", None)
    if isinstance(rem, list) and len(rem) > 0:
        if isinstance(rem[0], dict) and "machine_options" in rem[0]:
            return True
    return False


def _simulate_one_step(ready_operations, chosen_idx, time, machine_loads, machine_est_finish=None):
    """
    Simulate executing ready_operations[chosen_idx] on its best machine.
    Returns (estimated_makespan_component, new_machine_loads).
    Uses a smooth log-sum-exp surrogate for the max objective.
    """
    op = ready_operations[chosen_idx]
    mo = op.get("machine_options", {})
    durs = list(mo.values()) if mo else [0.0]
    
    # Pick the machine that minimises estimated completion time
    # (finish_time + duration), not just raw load.
    best_machine = None
    best_comp = float('inf')
    for m in mo.keys():
        if machine_est_finish is not None:
            start = machine_est_finish.get(m, time)
        else:
            start = time
        comp = start + mo[m]
        if comp < best_comp:
            best_comp = comp
            best_machine = m
    
    dur = mo.get(best_machine, min(durs)) if best_machine else (min(durs) if durs else 0.0)
    comp_time = best_comp
    
    # Build new machine loads: remove this op's load, add its duration
    new_loads = dict(machine_loads)
    # Remove this op from all machines it could use (it was counted)
    for m in mo.keys():
        if m in new_loads:
            new_loads[m] -= 1
    # Add the chosen machine's new load
    if best_machine is not None:
        new_loads[best_machine] = new_loads.get(best_machine, 0) + 1
    
    # Estimate new makespan component using smooth log-sum-exp
    # over: completion time of chosen op, and remaining work of others.
    max_rw_others = 0.0
    for j, other in enumerate(ready_operations):
        if j != chosen_idx:
            rw = other.get("remaining_work", 0.0)
            if rw > max_rw_others:
                max_rw_others = rw
    
    # Smooth max via log-sum-exp
    if _LSE_TEMP > 0:
        t = _LSE_TEMP
        vals = [comp_time, max_rw_others]
        m_val = max(vals)
        lse = m_val + t * np.log(sum(np.exp((v - m_val) / t) for v in vals))
    else:
        lse = max(comp_time, max_rw_others)
    
    return lse, new_loads


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard: if we're too close to the limit, fall back to
    # a single-pass heuristic immediately.
    if _remaining_time() < 0.5:
        return _fast_heuristic(ready_operations, time)
    
    machine_loads = _count_machine_loads(ready_operations)
    machine_est_finish = _estimate_machine_finish(ready_operations, time)
    n = len(ready_operations)
    
    # --- Score all ops ---
    scores = []
    for i, op in enumerate(ready_operations):
        s = _score_op(op, machine_loads, time, machine_est_finish)
        scores.append((s, i))
    
    # Sort descending by score
    scores.sort(reverse=True, key=lambda x: x[0])
    
    # If no look-ahead data, just pick the top one
    if not _can_look_ahead(ready_operations):
        return scores[0][1]
    
    # --- Beam search over top-K candidates ---
    top_k = min(_TOP_K, n)
    best_idx = scores[0][1]
    best_score = scores[0][0]
    
    # We do a shallow 1-step look-ahead for each top candidate.
    # The "score" we optimise is the heuristic score of the resulting
    # state, which we approximate as:
    #   (heuristic of chosen op) - (penalty for estimated makespan)
    
    # Pre-compute: for each candidate, what would the state look like?
    # We use a simple estimate: the max remaining work in the new state
    # minus the completed op's contribution.
    
    for k in range(top_k):
        idx = scores[k][1]
        est_mk, _ = _simulate_one_step(ready_operations, idx, time, machine_loads, machine_est_finish)
        
        # Composite: prefer high heuristic score, low estimated makespan
        base = scores[k][0]
        # Normalise est_mk roughly by time scale
        norm_mk = est_mk / (time + 1.0) if time > 0 else est_mk
        combined = base - 0.1 * norm_mk
        
        if combined > best_score:
            best_score = combined
            best_idx = idx
        
        # Deadline check inside loop
        if _remaining_time() < 1.0:
            break
    
    return best_idx


def _fast_heuristic(ready_operations, time):
    """
    Fast single-pass heuristic used when deadline is approaching.
    Uses the same smooth log-sum-exp makespan surrogate for consistency.
    """
    best_idx = 0
    best_score = float('-inf')
    
    # Quick machine load count
    loads = {}
    for op in ready_operations:
        for m in op.get("machine_options", {}):
            loads[m] = loads.get(m, 0) + 1
    
    # Estimate machine finish times (lightweight)
    est_finish = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        min_dur = min(mo.values())
        for m in mo:
            est_finish[m] = est_finish.get(m, 0.0) + min_dur
    for m in est_finish:
        est_finish[m] = time + est_finish[m]
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        mo = op.get("machine_options", {})
        durs = list(mo.values()) if mo else [0.0]
        min_dur = min(durs) if durs else 0.0
        
        # Min load among option machines
        min_load = min((loads.get(m, 0) for m in mo), default=0)
        
        # Estimate completion on best machine
        if mo:
            best_comp = min(est_finish.get(m, time) + mo[m] for m in mo)
        else:
            best_comp = time + min_dur
        
        # Smooth max via log-sum-exp
        if _LSE_TEMP > 0:
            t = _LSE_TEMP
            m_val = max(best_comp, rw, 0.0)
            lse = m_val + t * np.log(np.exp((best_comp - m_val) / t) + np.exp((rw - m_val) / t))
        else:
            lse = max(best_comp, rw)
        
        s = W_MWR * rw + W_SPT * (1.0 / (min_dur + 1.0)) - W_MACH * min_load - W_MK * lse
        if s > best_score:
            best_score = s
            best_idx = i
    
    return best_idx
