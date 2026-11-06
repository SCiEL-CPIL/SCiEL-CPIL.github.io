"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

_start_time = time.time()
TIME_BUDGET = 145.0  # safety margin before 150s kill


def _greedy_by_density(w, v, cap):
    """Greedy: pick items by value/weight ratio."""
    if len(w) == 0:
        return [], 0.0
    idx = np.argsort(-v / w)
    total_w = 0.0
    total_v = 0.0
    selected = []
    for i in idx:
        wi = w[i]
        if total_w + wi <= cap + 1e-9:
            total_w += wi
            total_v += v[i]
            selected.append(i)
    return selected, total_v


def _greedy_by_value(w, v, cap):
    """Greedy: pick items by value."""
    if len(w) == 0:
        return [], 0.0
    idx = np.argsort(-v)
    total_w = 0.0
    total_v = 0.0
    selected = []
    for i in idx:
        wi = w[i]
        if total_w + wi <= cap + 1e-9:
            total_w += wi
            total_v += v[i]
            selected.append(i)
    return selected, total_v


def _greedy_by_weight(w, v, cap):
    """Greedy: pick items by weight (lightest first)."""
    if len(w) == 0:
        return [], 0.0
    idx = np.argsort(w)
    total_w = 0.0
    total_v = 0.0
    selected = []
    for i in idx:
        wi = w[i]
        if total_w + wi <= cap + 1e-9:
            total_w += wi
            total_v += v[i]
            selected.append(i)
    return selected, total_v


def _local_search(w, v, cap, selected):
    """Try swapping each selected item with each unselected item to improve."""
    if len(w) == 0:
        return selected, 0.0
    selected_set = set(selected)
    unselected = [i for i in range(len(w)) if i not in selected_set]
    
    total_v = sum(v[i] for i in selected)
    total_w = sum(w[i] for i in selected)
    
    improved = True
    while improved:
        improved = False
        for si in list(selected):
            for ui in unselected:
                if ui in selected_set:
                    continue
                new_w = total_w - w[si] + w[ui]
                if new_w <= cap + 1e-9:
                    new_v = total_v - v[si] + v[ui]
                    if new_v > total_v + 1e-9:
                        selected.remove(si)
                        selected.append(ui)
                        selected_set.remove(si)
                        selected_set.add(ui)
                        total_w = new_w
                        total_v = new_v
                        improved = True
        # Also try adding items
        for ui in unselected:
            if ui in selected_set:
                continue
            new_w = total_w + w[ui]
            if new_w <= cap + 1e-9:
                new_v = total_v + v[ui]
                if new_v > total_v + 1e-9:
                    selected.append(ui)
                    selected_set.add(ui)
                    total_w = new_w
                    total_v = new_v
                    improved = True
    return selected, total_v


def _exhaustive_small(w, v, cap):
    """Exhaustive search for small number of feasible items."""
    n = len(w)
    if n == 0:
        return [], 0.0
    if n > 25:
        return None, None
    
    best_v = 0.0
    best_set = set()
    
    # Use recursive backtracking
    def backtrack(idx, curr_w, curr_v, curr_set):
        nonlocal best_v, best_set
        if curr_v > best_v + 1e-9:
            best_v = curr_v
            best_set = set(curr_set)
        if idx >= n:
            return
        # Option: skip
        backtrack(idx + 1, curr_w, curr_v, curr_set)
        # Option: take
        if curr_w + w[idx] <= cap + 1e-9:
            curr_set.append(idx)
            backtrack(idx + 1, curr_w + w[idx], curr_v + v[idx], curr_set)
            curr_set.pop()
    
    backtrack(0, 0.0, 0.0, [])
    return list(best_set), best_v


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if time.time() - _start_time > TIME_BUDGET:
        return -1
    
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    
    n = len(w)
    if n == 0 or cap <= 0:
        return -1
    
    # Filter to feasible items
    feasible_mask = w <= cap + 1e-9
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1
    
    fw = w[feasible_mask]
    fv = v[feasible_mask]
    n_feasible = len(fw)
    
    # If small enough, do exhaustive search
    if n_feasible <= 25:
        best_set, best_v = _exhaustive_small(fw, fv, cap)
        if best_set is not None and len(best_set) > 0:
            # Map back to original indices
            orig_indices = [feasible_indices[i] for i in best_set]
            # Pick the highest value item from the optimal set
            best_idx = max(orig_indices, key=lambda i: v[i])
            return int(best_idx)
        return -1
    
    # Otherwise, use multiple greedy + local search
    solutions = []
    
    # Greedy by density
    sol1, v1 = _greedy_by_density(fw, fv, cap)
    solutions.append((sol1, v1))
    
    # Greedy by value
    sol2, v2 = _greedy_by_value(fw, fv, cap)
    solutions.append((sol2, v2))
    
    # Greedy by weight
    sol3, v3 = _greedy_by_weight(fw, fv, cap)
    solutions.append((sol3, v3))
    
    # Local search on each
    best_solution = None
    best_value = -1.0
    
    for sol, val in solutions:
        if time.time() - _start_time > TIME_BUDGET:
            break
        improved_sol, improved_val = _local_search(fw, fv, cap, list(sol))
        if improved_val > best_value:
            best_value = improved_val
            best_solution = improved_sol
    
    if best_solution is None or len(best_solution) == 0:
        return -1
    
    # Map back to original indices
    orig_indices = [feasible_indices[i] for i in best_solution]
    # Pick the highest value item from the best solution
    best_idx = max(orig_indices, key=lambda i: v[i])
    return int(best_idx)
# EVOLVE-BLOCK-END
