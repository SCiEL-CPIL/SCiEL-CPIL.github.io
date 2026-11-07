"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill

    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # (5, 100) -> (100,)
    total_prize = prizes.sum(axis=0)

    # Per-constraint efficiency: prize / weight for each constraint
    # (5, 100)
    eff = prizes / (weights + 1e-9)

    # Geometric mean of per-constraint efficiencies:
    # Rewards items that are efficient across ALL constraints (bottleneck-aware)
    log_eff = np.log(eff + 1e-9)
    geo_mean = np.exp(log_eff.mean(axis=0))  # (100,)

    # Sum-ratio: total prize / total weight across all constraints
    # (100,)
    total_weight = weights.sum(axis=0)
    sum_ratio = total_prize / (total_weight + 1e-9)

    # Blend the two signals:
    # - geo_mean captures the "weakest link" (minimum constraint efficiency)
    # - sum_ratio captures overall efficiency
    # A multiplicative blend balances both; we normalize to keep scale comparable
    blend = geo_mean * sum_ratio

    # Ensure positive and finite
    blend = np.where(np.isfinite(blend) & (blend > 0), blend, 1e-9)

    # Check deadline (unlikely to be hit for simple arithmetic, but safe)
    if time.time() > deadline:
        return np.maximum(blend, 1e-9)

    return blend
# EVOLVE-BLOCK-END
