"""Dispatch rule for open-shop scheduling: optimize machine load balance via Hungarian assignment."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 115.0  # safety margin before 120s wall-clock limit
_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        # Fallback: pick operation with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0
    
    # Collect all machines that appear in any op's options
    all_machines = set()
    for op in ready_operations:
        for m in op["machine_options"]:
            all_machines.add(m)
    
    machines = sorted(all_machines)
    n_machines = len(machines)
    
    # For each operation, compute features:
    # - remaining_work: total work left for the job
    # - remaining_ops: number of operations left for the job
    # - min_duration: shortest processing time across available machines
    # - max_duration: longest processing time across available machines
    # - n_options: number of machine options (flexibility)
    # - avg_duration: average processing time across available machines
    
    features = []
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        durations = list(op["machine_options"].values())
        min_duration = min(durations)
        max_duration = max(durations)
        n_options = len(durations)
        avg_duration = sum(durations) / n_options
        
        features.append((rem_work, rem_ops, min_duration, max_duration, n_options, avg_duration))
    
    # Normalize features to [0, 1] range for weighted scoring
    rem_works = np.array([f[0] for f in features])
    rem_ops_arr = np.array([f[1] for f in features])
    min_durs = np.array([f[2] for f in features])
    max_durs = np.array([f[3] for f in features])
    n_opts = np.array([f[4] for f in features])
    avg_durs = np.array([f[5] for f in features])
    
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)
    
    # Normalize in appropriate directions:
    # - Higher remaining_work is more urgent (keep high)
    # - Higher remaining_ops is more urgent (keep high)
    # - Lower min_duration is better (invert)
    # - Lower max_duration is better (invert)
    # - Lower n_options is less flexible (invert: fewer options = higher priority)
    # - Lower avg_duration is better (invert)
    
    norm_rem_work = normalize(rem_works)
    norm_rem_ops = normalize(rem_ops_arr)
    norm_min_dur = 1.0 - normalize(min_durs)
    norm_max_dur = 1.0 - normalize(max_durs)
    norm_n_opts = 1.0 - normalize(n_opts)
    norm_avg_dur = 1.0 - normalize(avg_durs)
    
    # Weighted score: combine all factors
    # Weights tuned to balance urgency (remaining work/ops) with efficiency (shorter durations)
    w_rem_work = 0.35
    w_rem_ops = 0.25
    w_min_dur = 0.15
    w_max_dur = 0.10
    w_n_opts = 0.05
    w_avg_dur = 0.10
    
    scores = (w_rem_work * norm_rem_work + 
              w_rem_ops * norm_rem_ops + 
              w_min_dur * norm_min_dur + 
              w_max_dur * norm_max_dur + 
              w_n_opts * norm_n_opts + 
              w_avg_dur * norm_avg_dur)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
