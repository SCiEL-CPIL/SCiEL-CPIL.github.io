"""QAOA p=2 MaxCut angle design with graph-dependent optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = None


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _build_cost_matrix(n, edges):
    """Build the MaxCut cost diagonal as a vector of length 2^n."""
    size = 1 << n
    cost = np.zeros(size)
    for (u, v) in edges:
        # For each bitstring, check if u and v are on different sides
        mask_u = 1 << u
        mask_v = 1 << v
        for s in range(size):
            if ((s >> u) & 1) != ((s >> v) & 1):
                cost[s] += 1.0
    return cost


def _apply_x_rotation(state, beta):
    """Apply exp(-i beta * sum X) to state vector.
    
    exp(-i beta X) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    for each qubit.
    """
    n = int(np.log2(len(state)))
    result = state.copy()
    for q in range(n):
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        # Reshape to apply on qubit q
        shape = [1] * n
        shape[q] = 2
        # Split along qubit q
        dim = 1 << q
        period = dim * 2
        # For each block of size period, split into two halves of size dim
        new_result = np.zeros_like(result)
        for block_start in range(0, len(result), period):
            for i in range(dim):
                idx0 = block_start + i
                idx1 = block_start + dim + i
                new_result[idx0] = c * result[idx0] + s * result[idx1]
                new_result[idx1] = s * result[idx0] + c * result[idx1]
        result = new_result
    return result


def _expected_cut_value(n, edges, gammas, betas):
    """Compute expected cut for given QAOA angles."""
    size = 1 << n
    cost = _build_cost_matrix(n, edges)
    
    # Start in |+>^n
    state = np.ones(size, dtype=complex) / np.sqrt(size)
    
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i gamma * C): diagonal phase
        if abs(gamma) > 1e-12:
            state = state * np.exp(-1j * gamma * cost)
        # Apply exp(-i beta * sum X)
        if abs(beta) > 1e-12:
            state = _apply_x_rotation(state, beta)
    
    # Expected cut = sum |state_i|^2 * cost_i
    probs = np.abs(state) ** 2
    return np.real(np.dot(probs, cost))


def _exact_optimum(n, edges):
    """Compute exact MaxCut optimum by exhaustive search."""
    size = 1 << n
    best = 0
    for s in range(size):
        cut = 0
        for (u, v) in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best


def _optimize_angles(n, edges, p, time_budget):
    """Optimize QAOA angles for a given graph using grid search + local refinement."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_value = -1
    best_angles = np.array([[0.35, 0.25]] * p)
    
    # Grid search over gamma in [0, pi] and beta in [0, pi]
    # For p=2, we have 4 parameters. Use a coarse grid first.
    n_grid = 8  # per parameter
    
    gamma_grid = np.linspace(0, np.pi, n_grid)
    beta_grid = np.linspace(0, np.pi, n_grid)
    
    # Precompute cost matrix
    cost = _build_cost_matrix(n, edges)
    size = 1 << n
    
    # Coarse grid search
    for g1 in gamma_grid:
        for b1 in beta_grid:
            for g2 in gamma_grid:
                for b2 in beta_grid:
                    if time.time() > deadline:
                        break
                    gammas = [g1, g2]
                    betas = [b1, b2]
                    val = _expected_cut_value(n, edges, gammas, betas)
                    if val > best_value:
                        best_value = val
                        best_angles = np.array([[g1, b1], [g2, b2]])
    
    # Local refinement with finer grid around best
    if time.time() < deadline:
        g1_best, b1_best = best_angles[0]
        g2_best, b2_best = best_angles[1]
        
        # Refine each parameter
        for param_idx in range(4):
            for _ in range(3):  # 3 refinement rounds
                if time.time() > deadline:
                    break
                # Determine which parameter to refine
                if param_idx == 0:
                    center = best_angles[0, 0]
                elif param_idx == 1:
                    center = best_angles[0, 1]
                elif param_idx == 2:
                    center = best_angles[1, 0]
                else:
                    center = best_angles[1, 1]
                
                # Try perturbations
                for delta in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                    if time.time() > deadline:
                        break
                    test_angles = best_angles.copy()
                    if param_idx == 0:
                        test_angles[0, 0] = np.clip(center + delta, 0, np.pi)
                    elif param_idx == 1:
                        test_angles[0, 1] = np.clip(center + delta, 0, np.pi)
                    elif param_idx == 2:
                        test_angles[1, 0] = np.clip(center + delta, 0, np.pi)
                    else:
                        test_angles[1, 1] = np.clip(center + delta, 0, np.pi)
                    
                    val = _expected_cut_value(n, edges, test_angles[0, 0], test_angles[0, 1])
                    # Recompute properly
                    val = _expected_cut_value(n, edges, 
                                              [test_angles[0, 0], test_angles[1, 0]],
                                              [test_angles[0, 1], test_angles[1, 1]])
                    if val > best_value:
                        best_value = val
                        best_angles = test_angles
    
    return best_angles


def angles(n, edges, p):
    """Design QAOA angles for the given graph."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    remaining = _time_left()
    
    if remaining < 5.0:
        # Not enough time, return default
        return np.tile([0.35, 0.25], (p, 1))
    
    # Allocate time budget for optimization
    time_budget = max(2.0, remaining - 10.0)
    
    try:
        result = _optimize_angles(n, edges, p, time_budget)
        return result
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
