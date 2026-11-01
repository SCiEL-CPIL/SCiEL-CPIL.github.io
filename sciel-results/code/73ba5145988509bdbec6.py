import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145

    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    M_lb = total_diffs_needed

    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)

    if best_scope <= M_lb:
        return best_rows

    def solve_for_scope(M, time_limit):
        """
        Constraint-driven constructive search with dynamic variable ordering.
        Returns list of rows if found, else None.
        """
        rows = [[0] * (k + 1) for _ in range(n_rows)]
        assigned = [[False] * (k + 1) for _ in range(n_rows)]
        used_diffs = set()
        # List of (r, c, val) for assigned positions
        assigned_vals = [(r, 0, 0) for r in range(n_rows)]
        for r in range(n_rows):
            assigned[r][0] = True

        # Count of assigned positions (excluding the fixed 0s)
        # We need to assign n_rows * k positions total

        def backtrack(depth):
            if time.time() > time_limit:
                return False
            if depth == n_rows * k:
                return True

            # Find the most constrained unassigned position
            best_r, best_c = -1, -1
            best_cands = None
            best_len = float('inf')

            for r in range(n_rows):
                for c in range(1, k + 1):
                    if not assigned[r][c]:
                        # Quick bounds check
                        lower = 1
                        if c > 0 and assigned[r][c - 1]:
                            lower = rows[r][c - 1] + 1
                        remaining = k - c
                        upper = M - remaining
                        if lower > upper:
                            return False

                        # Get candidates
                        cands = []
                        for val in range(lower, upper + 1):
                            valid = True
                            for (r2, c2, v2) in assigned_vals:
                                d = abs(val - v2)
                                if d == 0:
                                    continue
                                if d in used_diffs:
                                    valid = False
                                    break
                            if valid:
                                cands.append(val)

                        if len(cands) < best_len:
                            best_len = len(cands)
                            best_cands = cands
                            best_r, best_c = r, c
                            if best_len == 0:
                                return False
                            if best_len == 1:
                                break
                if best_len <= 1:
                    break

            if best_r == -1:
                return True

            # Try each candidate
            for val in best_cands:
                rows[best_r][best_c] = val
                assigned[best_r][best_c] = True

                new_diffs = set()
                for (r2, c2, v2) in assigned_vals:
                    d = abs(val - v2)
                    if d > 0:
                        new_diffs.add(d)

                used_diffs.update(new_diffs)
                assigned_vals.append((best_r, best_c, val))

                if backtrack(depth + 1):
                    return True

                # Undo
                used_diffs.difference_update(new_diffs)
                assigned_vals.pop()
                assigned[best_r][best_c] = False
                rows[best_r][best_c] = 0

            return False

        if backtrack(0):
            return rows
        return None

    # Iterate over M values
    # To avoid spending too much time on infeasible small M, 
    # use a step and time-slicing approach
    remaining_time = deadline - time.time()
    if remaining_time < 3:
        return best_rows

    # Try M values, giving more time to larger M (more likely feasible)
    # Use a geometric-ish allocation
    M_start = M_lb
    M_end = best_scope

    if M_start >= M_end:
        return best_rows

    # Try a subset of M values to ensure we make progress
    num_to_try = min(M_end - M_start, 30)
    step = max(1, (M_end - M_start) // num_to_try)

    for M in range(M_start, M_end, step):
        if time.time() > deadline - 2:
            break

        remaining_Ms = max(1, (M_end - M) // step)
        t_slice = min(remaining_time / remaining_Ms, 10.0)
        t_limit = time.time() + t_slice

        sol = solve_for_scope(M, t_limit)
        if sol is not None:
            return sol

    return best_rows
