"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    n_resources = len(capacities)

    # 1. Calculate Resource Bottleneck Index for each activity
    # This measures the maximum fraction of any resource's capacity consumed by the activity.
    # High bottleneck means the activity is a resource constraint.
    # Shape: (n, n_resources)
    # Demand[i, r] / Capacity[r]
    # Handle zero capacity (should not happen in valid instances, but safe divide)
    safe_capacities = np.where(capacities == 0, 1e-9, capacities)
    utilization = demands / safe_capacities
    bottleneck = np.max(utilization, axis=1)  # Shape: (n,)

    # 2. Normalize Duration and Bottleneck to [0, 1]
    # Higher values should yield higher priority scores.
    
    dur_min, dur_max = durations.min(), durations.max()
    dur_range = dur_max - dur_min + 1e-9
    norm_duration = (durations - dur_min) / dur_range

    bott_min, bott_max = bottleneck.min(), bottleneck.max()
    bott_range = bott_max - bott_min + 1e-9
    norm_bottleneck = (bottleneck - bott_min) / bott_range

    # 3. Combine Heuristics
    # Duration is generally the primary driver for makespan.
    # Bottleneck is a secondary driver for resource contention.
    # Weights: Duration 0.7, Bottleneck 0.3
    w_duration = 0.7
    w_bottleneck = 0.3

    priority_score = w_duration * norm_duration + w_bottleneck * norm_bottleneck

    return priority_score
# EVOLVE-BLOCK-END
