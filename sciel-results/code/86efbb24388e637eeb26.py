"""Improved dispatch rule for Flexible Job-Shop Scheduling using machine-aware selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0
_T0 = None

def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    return time.perf_counter() - _T0

def dispatch(ready_operations, time):
    if _elapsed() > _DEADLINE:
        if not ready_operations:
            return 0
        # Fallback: Pick op with max remaining work, min duration machine
        idx = int(np.argmax([op["remaining_work"] for op in ready_operations]))
        op = ready_operations[idx]
        mo = op["machine_options"]
        if mo:
            best_m = min(mo, key=mo.get)
            return (idx, best_m)
        return idx

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        mo = op["machine_options"]
        if mo:
            best_m = min(mo, key=mo.get)
            return (0, best_m)
        return 0

    # Calculate scores
    # Heuristic: Prioritize jobs with more remaining work, but penalize those with many remaining ops (low urgency)
    # Actually, high urgency (few remaining ops) is often better to finish jobs early and free resources.
    # Let's use a combination: Score = w1 * (remaining_work / max_work) + w2 * (1 / remaining_ops)
    
    works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    max_work = np.max(works)
    if max_work == 0:
        max_work = 1.0
    
    work_norm = works / max_work
    # Urgency: 1/ops. If ops=1, urgency=1. If ops=10, urgency=0.1.
    urgency = 1.0 / np.maximum(ops, 1.0)
    max_urgency = np.max(urgency)
    if max_urgency == 0:
        max_urgency = 1.0
    urgency_norm = urgency / max_urgency

    # Weights: Balance between work volume and urgency
    w_work = 0.6
    w_urgency = 0.4
    
    scores = w_work * work_norm + w_urgency * urgency_norm
    
    # Select best operation
    idx = int(np.argmax(scores))
    
    # Select best machine for this operation (minimum duration)
    op = ready_operations[idx]
    mo = op["machine_options"]
    if mo:
        # Choose machine with shortest processing time
        best_m = min(mo, key=mo.get)
        return (idx, best_m)
    else:
        return idx
# EVOLVE-BLOCK-END
