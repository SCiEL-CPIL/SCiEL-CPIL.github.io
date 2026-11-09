"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: keep a safety margin before the 120s (or 150s) limit.
# We track cumulative CPU time spent in dispatch to avoid pathological loops.
_dispatch_start_wall = _time.time()
_MAX_WALL_SECONDS = 110.0  # safety margin
_dispatch_call_count = 0


def dispatch(ready_operations, time):
    global _dispatch_call_count
    _dispatch_call_count += 1

    # Fast deadline check
    if _time.time() - _dispatch_start_wall > _MAX_WALL_SECONDS:
        # Fallback: pick the operation with the most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0

    # Precompute features for each ready operation
    n = len(ready_operations)

    # For each ready operation, we need to estimate the impact of starting it now.
    # Key features:
    # 1. remaining_work: total work left on this job (urgency)
    # 2. remaining_ops: number of operations left (urgency)
    # 3. min_duration: shortest possible processing time for this op (to free up machine quickly)
    # 4. max_duration: longest possible processing time
    # 5. num_machines: how many machines can process this op
    # 6. machine_load: estimated load on the best machine for this op

    # We'll compute a score for each operation. Higher score = better to start.
    scores = np.zeros(n)

    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]

        durations = list(machine_options.values())
        machines = list(machine_options.keys())

        min_dur = min(durations)
        max_dur = max(durations)
        num_machines = len(machines)

        # Urgency: jobs with less remaining work/ops are closer to completion.
        # But we want to prioritize jobs that, if we don't start them, will block others.
        # Actually, in open-shop, each job must visit each machine exactly once.
        # remaining_work is the sum of durations of all remaining ops for this job.
        # remaining_ops is the count of remaining ops.

        # Heuristic 1: Start the operation with the longest min_duration on its best machine?
        # No, we want to balance machine loads.

        # Let's estimate machine loads. We don't have direct access to machine loads,
        # but we can infer from the set of ready operations.
        # Actually, the ready_operations are those whose prerequisites are met.
        # In open-shop, an operation is ready if the job hasn't visited that machine yet
        # and the machine is free? No, in open-shop, the constraint is that each job visits
        # each machine at most once, and each machine processes at most one job at a time.
        # So ready operations are those where the job hasn't been processed on that machine yet,
        # AND the machine is currently free.

        # Since we don't have machine load info directly, let's use a simpler heuristic.
        # The key insight from the "load-aware" finding: balance machine congestion and job urgency.

        # Let's estimate the "cost" of starting this op:
        # - If we start it, the machine will be busy for min_dur (best case) to max_dur.
        # - We prefer ops that can be done quickly on a machine that isn't heavily loaded.

        # Since we can't directly measure machine load, let's use a proxy:
        # Count how many other ready ops could use the same machine.
        # This is a local congestion measure.

        # For each machine that can process this op, count how many other ready ops
        # also list that machine as an option.
        machine_congestion = {}
        for m in machines:
            count = 0
            for j, other_op in enumerate(ready_operations):
                if j == i:
                    continue
                if m in other_op["machine_options"]:
                    count += 1
            machine_congestion[m] = count

        # The best machine for this op is the one with lowest congestion.
        best_machine = min(machines, key=lambda m: (machine_congestion[m], machine_options[m]))
        best_dur = machine_options[best_machine]
        best_congestion = machine_congestion[best_machine]

        # Score components:
        # 1. Urgency: higher remaining_work and remaining_ops mean more critical.
        #    But actually, we want to finish jobs, so higher remaining_work = more urgent.
        #    Wait, if a job has high remaining_work, it has a lot left. We might want to
        #    start ops that are "close to done" to free up jobs.
        #    Actually, in scheduling, we often use Longest Remaining Processing Time (LRPT)
        #    for jobs. But here we're choosing which op to start.

        # Let's think differently. The makespan is determined by the last job to finish.
        # To minimize makespan, we want to avoid idle time and balance loads.

        # Heuristic: Start the op that minimizes the expected completion time of its job,
        # while also considering machine balance.

        # Simple score:
        # - Prefer ops with shorter processing time on the least congested machine.
        # - Prefer jobs with more remaining work (to keep them moving).

        # Let's try: score = -best_dur * (1 + best_congestion * 0.5) + remaining_work * 0.1
        # This prefers short durations on less congested machines, with a small boost for urgent jobs.

        score = -best_dur * (1.0 + best_congestion * 0.5) + remaining_work * 0.1 + remaining_ops * 0.5

        scores[i] = score

    # Pick the operation with the highest score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
