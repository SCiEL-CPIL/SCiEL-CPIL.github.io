"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Compute cumulative load per machine: sum of minimum durations across all ready ops
    # that can use each machine. This gives a real-time congestion estimate.
    machine_cumulative_load = {}
    all_ops_info = []

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        job_remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)

        for machine, duration in machine_options.items():
            if machine not in machine_cumulative_load:
                machine_cumulative_load[machine] = 0
            machine_cumulative_load[machine] += duration

        all_ops_info.append({
            'index': i,
            'job_remaining_work': job_remaining_work,
            'remaining_ops': remaining_ops,
            'machine_options': machine_options
        })

    if not all_ops_info:
        return 0

    # Score each operation: maximize (job critical path) / (estimated cumulative load on best machine)
    # We pick the machine that minimizes the denominator for each op, then compare ops.
    best_score = -1
    best_index = 0
    best_machine = None

    for info in all_ops_info:
        i = info['index']
        job_remaining_work = info['job_remaining_work']
        remaining_ops = info['remaining_ops']
        machine_options = info['machine_options']

        # Estimate critical path more robustly:
        # Use remaining_work (sum of min durations of remaining ops) as base.
        # Add a small penalty for many remaining ops to account for sequencing constraints.
        criticality = job_remaining_work
        if remaining_ops > 1:
            criticality *= (1.0 + 0.05 * (remaining_ops - 1))

        # For each eligible machine, compute the score using cumulative load
        best_op_score = -1
        best_op_machine = None

        for machine, duration in machine_options.items():
            # Cumulative load on this machine from all ready ops
            cum_load = machine_cumulative_load.get(machine, duration)

            # The effective time contribution if we schedule this op on this machine:
            # It adds 'duration' to the machine's load, and the job's critical path is reduced.
            # We want to minimize the ratio of job criticality to machine availability.
            # Score = criticality / (cum_load + duration)
            # Higher score means more urgent relative to machine congestion.

            effective_denominator = cum_load + duration
            if effective_denominator <= 0:
                effective_denominator = 1e-9

            score = criticality / effective_denominator

            if score > best_op_score:
                best_op_score = score
                best_op_machine = machine

        if best_op_machine is None:
            continue

        if best_op_score > best_score:
            best_score = best_op_score
            best_index = i
            best_machine = best_op_machine

    return (best_index, best_machine)
# EVOLVE-BLOCK-END
