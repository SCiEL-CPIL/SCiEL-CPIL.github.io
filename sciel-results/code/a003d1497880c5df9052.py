import numpy as np
import time as time_module
from collections import defaultdict

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: 150s total, use 140s for dispatch
    deadline = time_module.time() + 140.0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Precompute features
    features = []
    for op in ready_operations:
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0
        avg_dur = sum(durations) / len(durations) if durations else 0
        n_machines = len(durations) if durations else 1
        rem_ops = op.get("remaining_ops", [])
        n_rem = len(rem_ops) if isinstance(rem_ops, list) else 0
        rw = op["remaining_work"]

        features.append({
            "remaining_work": rw,
            "min_dur": min_dur,
            "max_dur": max_dur,
            "avg_dur": avg_dur,
            "n_machines": n_machines,
            "n_rem_ops": n_rem,
        })

    # Per-job remaining work
    job_rw = defaultdict(float)
    for i, op in enumerate(ready_operations):
        jid = op["job"]
        job_rw[jid] = max(job_rw[jid], op["remaining_work"])

    # Check if simulation is possible
    can_simulate = False
    sample_op = ready_operations[0]
    rem_ops = sample_op.get("remaining_ops", None)
    if isinstance(rem_ops, list) and len(rem_ops) > 0:
        if isinstance(rem_ops[0], dict) and "machine_options" in rem_ops[0]:
            can_simulate = True

    # Smooth max via log-sum-exp
    def soft_max(values, beta=15.0):
        if not values:
            return 0.0
        arr = np.array(values, dtype=np.float64)
        m = np.max(arr)
        if m == 0:
            return 0.0
        return float(m + np.log(np.sum(np.exp(beta * (arr - m)))) / beta)

    def estimate_makespan(ready_ops, current_time):
        if not ready_ops:
            return current_time
        jr = defaultdict(float)
        for op in ready_ops:
            jid = op["job"]
            jr[jid] = max(jr[jid], op["remaining_work"])
        if jr:
            works = list(jr.values())
            return current_time + soft_max(works, beta=15.0)
        return current_time

    if can_simulate and n <= 12:
        def simulate_step(ready_ops, t, idx):
            op = ready_ops[idx]
            durations = list(op["machine_options"].values())
            min_dur = min(durations) if durations else 0
            comp_t = t + min_dur
            new_ready = []
            for j, other in enumerate(ready_ops):
                if j == idx:
                    rem = op.get("remaining_ops", [])
                    if isinstance(rem, list) and len(rem) > 0:
                        nxt = rem[0].copy()
                        nxt["remaining_work"] = max(0, op["remaining_work"] - min_dur)
                        nxt["remaining_ops"] = rem[1:] if len(rem) > 1 else []
                        new_ready.append(nxt)
                else:
                    new_ready.append(other)
            return new_ready, comp_t

        beam_width = 5
        candidates = []
        for i in range(n):
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_step(ready_operations, time, i)
            score = estimate_makespan(new_ready, new_t)
            candidates.append((score, i))

        if not candidates:
            return int(np.argmax([f["remaining_work"] for f in features]))

        candidates.sort(key=lambda x: x[0])
        top_k = candidates[:beam_width]

        best_score = float('inf')
        best_idx = top_k[0][1]

        for l1_score, l1_idx in top_k:
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_step(ready_operations, time, l1_idx)
            for j in range(len(new_ready)):
                if time_module.time() > deadline:
                    break
                nr2, nt2 = simulate_step(new_ready, new_t, j)
                s2 = estimate_makespan(nr2, nt2)
                total = soft_max([l1_score, s2], beta=20.0)
                if total < best_score:
                    best_score = total
                    best_idx = l1_idx

        return best_idx

    # Fallback: weighted heuristic
    weight_sets = [
        (1.0, 0.1, 0.5, 0.3),
        (1.0, 0.0, 0.5, 0.0),
        (0.8, 0.2, 0.3, 0.5),
        (1.0, 0.05, 0.2, 0.4),
        (0.5, 0.3, 0.5, 0.5),
        (1.0, 0.15, 0.0, 0.5),
        (0.9, 0.1, 0.4, 0.4),
        (1.0, 0.05, 0.3, 0.3),
    ]

    best_overall = float('inf')
    best_choice = 0

    for w1, w2, w3, w4 in weight_sets:
        if time_module.time() > deadline:
            break
        scores = []
        for f in features:
            s = (w1 * f["remaining_work"]
                 + w2 * (1.0 / (f["min_dur"] + 1.0)) * f["remaining_work"]
                 + w3 * f["n_rem_ops"] * 10.0
                 + w4 * f["remaining_work"] / max(f["n_machines"], 1))
            scores.append(s)
        chosen = int(np.argmax(scores))
        chosen_min = features[chosen]["min_dur"]
        comp_t = time + chosen_min
        chosen_job = ready_operations[chosen]["job"]
        chosen_rw_after = max(0, features[chosen]["remaining_work"] - chosen_min)
        jwa = {}
        for i, f in enumerate(features):
            jid = ready_operations[i]["job"]
            if i == chosen:
                jwa[jid] = chosen_rw_after
            else:
                if jid in jwa:
                    jwa[jid] = max(jwa[jid], f["remaining_work"])
                else:
                    jwa[jid] = f["remaining_work"]
        if jwa:
            est = comp_t + soft_max(list(jwa.values()), beta=15.0)
        else:
            est = comp_t
        if est < best_overall:
            best_overall = est
            best_choice = chosen

    return best_choice
