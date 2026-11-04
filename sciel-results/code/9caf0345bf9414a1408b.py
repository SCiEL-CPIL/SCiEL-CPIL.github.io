"""Difference triangle set (4,7) – optimized search."""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Precompute total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # --- Helper functions ---
    def all_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs

    def max_entry(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        diffs = all_diffs(rows)
        return len(diffs) == len(set(diffs))

    def compute_violations(rows):
        """Return number of duplicate differences (excess count)."""
        diffs = all_diffs(rows)
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        return sum(c - 1 for c in counts.values() if c > 1)

    def compute_cost(rows):
        """Cost = violations * 10000 + max_entry."""
        v = compute_violations(rows)
        m = max_entry(rows)
        return v * 10000 + m

    # --- Phase 1: Greedy initialization ---
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # --- Phase 1b: Randomized greedy for diverse starts ---
    def randomized_greedy(seed_val):
        rng = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    # With some probability, skip this candidate to explore
                    if rng.random() < 0.15 and len(row) < k:
                        continue
                    row.append(x)
                    used |= new
            # If we didn't fill the row, fill greedily
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
        return rows

    # --- Phase 2: Simulated annealing ---
    def simulated_annealing(initial_rows, time_budget):
        current_rows = [r[:] for r in initial_rows]
        current_cost = compute_cost(current_rows)
        best_rows = [r[:] for r in initial_rows]
        best_cost = current_cost
        best_max = max_entry(best_rows)

        # Indices to perturb: (row_idx, pos_idx) where pos_idx in 1..k
        indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]

        T = 80.0
        T_min = 0.05
        cooling = 0.99998

        end_time = time.time() + time_budget
        while time.time() < end_time:
            if T < T_min:
                # Restart with higher temperature occasionally
                if random.random() < 0.1:
                    T = 30.0
                else:
                    break

            # Pick a random perturbation
            r, i = random.choice(indices)
            old_val = current_rows[r][i]

            # Determine valid range for new value
            lower = current_rows[r][i - 1] + 1
            upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 30

            if lower > upper:
                continue

            # Bias towards smaller values
            if random.random() < 0.4:
                span = upper - lower + 1
                new_val = lower + int(random.random() * span * 0.6)
            else:
                new_val = random.randint(lower, upper)

            if new_val == old_val:
                continue

            # Apply move
            current_rows[r][i] = new_val

            # Check strict increasing
            row = current_rows[r]
            valid_increasing = True
            for j in range(1, len(row)):
                if row[j] <= row[j - 1]:
                    valid_increasing = False
                    break

            if not valid_increasing:
                current_rows[r][i] = old_val
                continue

            new_cost = compute_cost(current_rows)

            # Simulated annealing acceptance
            delta = new_cost - current_cost
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                new_max = max_entry(current_rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [rr[:] for rr in current_rows]
                    best_cost = current_cost
            else:
                current_rows[r][i] = old_val

            T *= cooling

        return best_rows, best_max

    # --- Phase 3: Local search (hill climbing on valid solutions) ---
    def local_search(rows, time_budget):
        current_rows = [r[:] for r in rows]
        current_max = max_entry(current_rows)
        best_rows = [r[:] for r in rows]
        best_max = current_max

        indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
        end_time = time.time() + time_budget

        while time.time() < end_time:
            improved = False
            for r, i in indices:
                old_val = current_rows[r][i]
                lower = current_rows[r][i - 1] + 1
                upper = min(current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 20,
                            current_max - 1)

                if lower > upper:
                    continue

                # Try to reduce the value
                for new_val in range(lower, old_val):
                    current_rows[r][i] = new_val
                    row = current_rows[r]
                    if any(row[j] <= row[j - 1] for j in range(1, len(row))):
                        current_rows[r][i] = old_val
                        break

                    if is_valid(current_rows):
                        new_max = max_entry(current_rows)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [rr[:] for rr in current_rows]
                            current_max = new_max
                            improved = True
                        else:
                            current_rows[r][i] = old_val
                        break
                    else:
                        current_rows[r][i] = old_val
                        break

                if improved:
                    break

            if not improved:
                break

        return best_rows, best_max

    # --- Main search loop ---
    best_rows = greedy_init()
    best_max = max_entry(best_rows)

    # Try multiple randomized greedy starts
    for seed_val in range(10):
        if time.time() > deadline:
            break
        rows = randomized_greedy(seed_val)
        if is_valid(rows):
            m = max_entry(rows)
            if m < best_max:
                best_max = m
                best_rows = rows

    # Run simulated annealing from best start
    remaining = deadline - time.time()
    if remaining > 5:
        sa_rows, sa_max = simulated_annealing(best_rows, remaining * 0.6)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows

    # Run local search
    remaining = deadline - time.time()
    if remaining > 5:
        ls_rows, ls_max = local_search(best_rows, remaining * 0.7)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    # Final check
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_init()
        best_max = max_entry(best_rows)

    return best_rows

# EVOLVE-BLOCK-END
