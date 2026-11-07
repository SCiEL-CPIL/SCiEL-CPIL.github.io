import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290.0

    try:
        prizes = np.asarray(prizes, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if prizes.shape != (5, 100) or weights.shape != (5, 100):
            raise ValueError("Unexpected shape")

        n_constraints, n_items = 5, 100
        capacity = 25.0
        eps = 1e-9

        # Per-constraint scarcity: how "tight" each constraint is
        # Higher = more scarce = more important to optimize for
        total_w = np.sum(weights, axis=1)  # (5,)
        scarcity = total_w / capacity      # (5,)

        # Per-item, per-constraint efficiency: prize / weight
        safe_w = weights + eps
        ratios = prizes / safe_w  # (5, 100)

        # Bottleneck-weighted efficiency
        # In scarce constraints, efficiency matters more
        # Use log to avoid extreme values, then exponentiate
        log_ratios = np.log(ratios + eps)  # (5, 100)
        scarcity_weights = scarcity / (np.sum(scarcity) + eps)  # normalize to sum to 1
        weighted_log_eff = np.sum(scarcity_weights[:, np.newaxis] * log_ratios, axis=0)  # (100,)
        bottleneck_eff = np.exp(weighted_log_eff)  # (100,)

        # Geometric slack: reward items that use less of each resource
        # In bottleneck constraints, slack matters more
        slack = 1.0 - weights / capacity  # (5, 100)
        slack = np.clip(slack, eps, 1.0)
        log_slack = np.log(slack)  # (5, 100)
        weighted_log_slack = np.sum(scarcity_weights[:, np.newaxis] * log_slack, axis=0)  # (100,)
        bottleneck_slack = np.exp(weighted_log_slack)  # (100,)

        # Combine: efficiency is the primary driver, slack is secondary
        # Use a product to reward items that are both efficient AND leave room
        score = np.power(bottleneck_eff + eps, 1.2) * np.power(bottleneck_slack + eps, 0.3)

        # Moderate sharpening to make ACO more selective
        heuristic = np.power(score + eps, 1.5)

        # Cleanup: ensure positive, finite, correct shape
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)
        heuristic = heuristic.reshape(n_items)

        if time.time() > deadline:
            return heuristic

        return heuristic

    except Exception:
        return np.full((100,), 1e-9)
