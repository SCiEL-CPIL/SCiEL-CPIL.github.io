import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import time

def construct():
    start_time = time.time()
    timeout = 75  # safety margin before 90s kill

    def is_valid(subset):
        """Check if subset has no sub-subset summing to 200."""
        if not subset:
            return True
        sums = {0}
        for x in subset:
            new_sums = {s + x for s in sums if s + x <= 200}
            if 200 in new_sums:
                return False
            sums |= new_sums
        return True

    def greedy_fallback():
        """Greedy from top down, avoiding subset sum 200."""
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    best_result = greedy_fallback()
    best_card = len(best_result)
    
    # Verify initial best result (should be valid by construction)
    if not is_valid(best_result):
        # This should never happen, but fallback to empty or simple set if logic breaks
        return [100] 

    # Try multiple lower bounds L for the interval [L, 100]
    # Larger L means fewer elements but denser valid subsets
    # We want to maximize cardinality, so we try various L values

    for L in [40, 50, 60, 70, 80, 90]:
        if time.time() - start_time > timeout:
            break

        elements = list(range(L, 101))
        n = len(elements)
        idx_map = {val: i for i, val in enumerate(elements)}

        # If this interval is smaller than our best, skip
        if n <= best_card:
            continue

        # Generate constraints: subsets summing to 200
        # For distinct elements in [L,100], pairs max sum is 99+100=199 < 200
        # So start from k=3
        # Find max_k: smallest k such that min sum of k elements > 200
        max_k = 1
        while max_k < n:
            min_sum = max_k * L + max_k * (max_k - 1) // 2
            if min_sum > 200:
                max_k -= 1
                break
            max_k += 1
        max_k = min(max_k, 6)  # cap for safety

        constraints = []

        # Generate all combinations of size 3 to max_k that sum to 200
        for k in range(3, max_k + 1):
            for comb in itertools.combinations(elements, k):
                if sum(comb) == 200:
                    row = [0] * n
                    for val in comb:
                        row[idx_map[val]] = 1
                    constraints.append(row)

            # Time check
            if time.time() - start_time > timeout:
                break

        if time.time() - start_time > timeout:
            break

        # If no constraints found, we can take all elements
        if not constraints:
            candidate = elements
            if is_valid(candidate) and len(candidate) > best_card:
                return candidate
            continue

        # Build ILP
        # Maximize sum(x)
        c = -np.ones(n)

        A_ub = np.array(constraints, dtype=np.float64)
        b_ub = np.ones(len(constraints)) * 2  # Sum <= 2 (at most 2 from each forbidden subset)

        # x_i in {0, 1}
        integrality = np.ones(n)
        bounds = Bounds(lb=0, ub=1)

        # Solve
        try:
            remaining = timeout - (time.time() - start_time)
            if remaining < 5:
                break
            time_limit = max(5, remaining - 5)
            res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)],
                       integrality=integrality, bounds=bounds,
                       options={"time_limit": time_limit})

            if res.status == 0:
                selected_indices = np.where(res.x > 0.5)[0]
                selected_values = [elements[i] for i in selected_indices]

                # Local search polish: try adding elements that don't create a forbidden subset
                # and removing elements that allow adding more
                improved = True
                while improved and (time.time() - start_time) < timeout:
                    improved = False
                    sel_set = set(selected_values)
                    # Try to add each excluded element
                    for x in elements:
                        if x in sel_set:
                            continue
                        # Check if adding x creates a subset summing to 200
                        # Compute all subset sums of current selection that are <= 200 - x
                        current_sums = {0}
                        for v in selected_values:
                            current_sums = {s + v for s in current_sums if s + v <= 200 - x}
                        if 200 - x not in current_sums:
                            selected_values.append(x)
                            sel_set.add(x)
                            improved = True
                            break
                    # Try to remove an element and add two others
                    for i, x in enumerate(list(selected_values)):
                        candidate = selected_values[:i] + selected_values[i+1:]
                        cand_set = set(candidate)
                        # Try to add two elements not in cand_set
                        added = 0
                        for y in elements:
                            if y in cand_set:
                                continue
                            # Check if adding y is safe
                            current_sums = {0}
                            for v in candidate:
                                current_sums = {s + v for s in current_sums if s + v <= 200 - y}
                            if 200 - y not in current_sums:
                                candidate.append(y)
                                cand_set.add(y)
                                added += 1
                                if added == 2:
                                    break
                        if added == 2:
                            selected_values = candidate
                            improved = True
                            break

                # Verify the result before updating best
                if is_valid(selected_values):
                    if len(selected_values) > best_card:
                        best_result = selected_values
                        best_card = len(best_result)
            else:
                # ILP failed, try greedy on this interval
                selected, reachable = [], {0}
                for x in reversed(elements):
                    new = {s + x for s in reachable if s + x <= 200}
                    if 200 not in new:
                        selected.append(x)
                        reachable |= new
                if is_valid(selected) and len(selected) > best_card:
                    best_result = selected
                    best_card = len(best_result)
        except Exception:
            # Fallback to greedy on this interval
            selected, reachable = [], {0}
            for x in reversed(elements):
                new = {s + x for s in reachable if s + x <= 200}
                if 200 not in new:
                    selected.append(x)
                    reachable |= new
            if is_valid(selected) and len(selected) > best_card:
                best_result = selected
                best_card = len(best_result)

    # Final safety check
    if not is_valid(best_result):
        return greedy_fallback()

    return best_result
