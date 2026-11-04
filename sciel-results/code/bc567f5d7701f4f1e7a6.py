"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard: kill before 150s wall-clock limit
    deadline = time.time() + 145.0

    # --- Helper: compute all positive differences for a list of rows ---
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

    # --- Phase 1: Greedy initialization (same as seed) ---
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

    best_rows = greedy_init()
    best_max = max_entry(best_rows)

    # --- Phase 2: Simulated annealing to reduce the largest entry ---
    current_rows = [r[:] for r in best_rows]
    current_max = best_max

    # Precompute the set of "indices" we can perturb: (row_idx, pos_idx) where pos_idx in 1..k
    # We never change position 0 (always 0)
    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]

    T = 50.0
    T_min = 0.01
    cooling = 0.99995

    # For efficiency, maintain the set of used differences incrementally is complex.
    # Instead, we compute the violation cost which is:
    #   cost = (number of duplicate differences) * 10000 + max_entry
    # But recomputing all diffs every step is O(n_rows * k^2).
    # For k=8, n_rows=4, that's 4*28 = 112 diffs. Very fast.

    def compute_cost(rows):
        diffs = all_diffs(rows)
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        violations = sum(c - 1 for c in counts.values() if c > 1)
        m = max_entry(rows)
        return violations * 10000 + m

    current_cost = compute_cost(current_rows)
    # We know current is valid, so violations=0, cost = current_max

    while time.time() < deadline:
        if T < T_min:
            break

        # Pick a random perturbation
        r, i = random.choice(indices)
        old_val = current_rows[r][i]

        # Determine valid range for new value
        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 50

        # Bias towards smaller values to reduce max
        # Use a distribution that favors values near the lower end
        if random.random() < 0.3:
            # Try to shrink: pick a value closer to lower
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.5)
        else:
            # Random in range
            new_val = random.randint(lower, upper)

        if new_val == old_val:
            continue

        # Apply move
        current_rows[r][i] = new_val

        # Check strict increasing is still maintained (should be by construction)
        # But let's verify for safety
        row = current_rows[r]
        if any(row[j] <= row[j-1] for j in range(1, len(row))):
            current_rows[r][i] = old_val
            continue

        new_cost = compute_cost(current_rows)

        # Simulated annealing acceptance
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_cost = new_cost
            new_max = max_entry(current_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [rr[:] for rr in current_rows]

        else:
            # Reject
            current_rows[r][i] = old_val

        T *= cooling

    return best_rows
# EVOLVE-BLOCK-END
