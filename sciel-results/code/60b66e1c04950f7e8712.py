import time
import random
import sys
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    # We'll use a set of used differences for clarity, but for speed we can use a set of ints
    # or a boolean array. Since scope is at most a few hundred to a few thousand,
    # a set of ints is fine.

    def get_diffs(row):
        """Get all differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_construct(scope_limit=None):
        """
        Greedy construction: build rows one by one, each row greedily.
        Returns list of rows or None if stuck.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try to add next element
                found = False
                # Start from last_val + 1, go up
                last_val = row[-1]
                # If scope_limit, don't exceed it
                max_val = scope_limit if scope_limit is not None else 10000
                for x in range(last_val + 1, max_val + 1):
                    # Check if x can be added
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        # Update used_diffs with new differences
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing to reduce scope while maintaining validity.
        Uses a weighted greedy coloring-inspired heuristic: prioritize moving
        elements that have high "degree" (many conflicting differences) and
        prefer smaller new values to minimize max label.
        """
        rows = [r[:] for r in initial_rows]
        
        # Compute current used_diffs and scope
        def compute_state(rows):
            used_diffs = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        used_diffs.add(row[j] - row[i])
            scope = max(max(r) for r in rows)
            return used_diffs, scope

        used_diffs, current_scope = compute_state(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Precompute row difference counts for "degree" heuristic
        # Degree of a position = number of differences it participates in
        # For position pos in a row of length k+1, degree = k (all other positions)
        # But we want to weight by how "constrained" it is - positions with
        # many small differences are more constrained

        def position_degree(row, pos):
            """Compute a heuristic degree for a position based on its differences."""
            degree = 0
            for i in range(k + 1):
                if i != pos:
                    d = abs(row[pos] - row[i])
                    # Smaller differences are more "expensive" (more likely to conflict)
                    # So weight by 1/d to prioritize positions with many small diffs
                    degree += 1.0 / d
            return degree

        # Temperature schedule
        t = start_time
        initial_temp = max(1.0, current_scope / 10.0)
        cooling_rate = 0.9995  # Slower cooling for more exploration

        steps_since_improvement = 0
        max_steps_without_improvement = 500

        # For scope computation optimization
        row_maxes = [max(r) for r in rows]

        while time.time() < t + time_limit:
            # Pick a random row and a random position to modify
            # Weighted selection: positions with higher degree are more likely to be chosen
            row_idx = random.randint(0, n_rows - 1)
            
            # Compute degrees for all positions in the selected row
            row = rows[row_idx]
            degrees = [0.0]  # Position 0 is fixed
            for pos in range(1, k + 1):
                degrees.append(position_degree(row, pos))
            
            # Normalize degrees to get probabilities
            total_deg = sum(degrees[1:])
            if total_deg == 0:
                pos = random.randint(1, k)
            else:
                # Weighted random selection
                r = random.random() * total_deg
                cumulative = 0.0
                pos = k
                for p in range(1, k + 1):
                    cumulative += degrees[p]
                    if r <= cumulative:
                        pos = p
                        break

            old_val = rows[row_idx][pos]
            # Try to change this value
            # New value should be > previous element and < next element (if exists)
            min_val = rows[row_idx][pos-1] + 1
            max_val = rows[row_idx][pos+1] - 1 if pos < k else best_scope + 10
            
            if min_val > max_val:
                continue

            # Prefer smaller new values (greedy coloring heuristic)
            # Bias towards values closer to min_val
            if max_val - min_val > 50:
                # Use a power-law distribution biased towards smaller values
                bias = random.random() ** 0.7
                new_val = min_val + int(bias * (max_val - min_val))
            else:
                new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue

            # Check if changing this value maintains validity
            # We need to:
            # 1. Remove old differences involving rows[row_idx][pos]
            # 2. Add new differences
            # 3. Check for conflicts

            # Collect old diffs involving position pos
            old_diffs = set()
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.add(d)

            # Temporarily remove old_diffs from used_diffs
            for d in old_diffs:
                used_diffs.discard(d)

            # Try adding new diffs
            new_diffs = set()
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)

            if not valid:
                # Restore old_diffs
                used_diffs.update(old_diffs)
                steps_since_improvement += 1
                continue

            # Compute new scope using precomputed row_maxes
            new_row_max = max(rows[row_idx][pos-1], new_val, rows[row_idx][pos+1] if pos < k else 0)
            # Actually, new_row_max should be the max of the entire modified row
            # Since only one element changes, new_row_max = max(old_row_max, new_val) if new_val > old_val, else old_row_max
            # But wait, if new_val < old_val, the row max might decrease
            # Let's compute it properly: the row is rows[row_idx] with pos set to new_val
            # The max of the row is max of all elements except we replace old_val with new_val
            # Since the row is strictly increasing, the max is always the last element
            # So if pos == k, new_row_max = new_val
            # Otherwise, new_row_max = row_maxes[row_idx] (unchanged, since pos < k means last element unchanged)
            
            if pos == k:
                new_row_max = new_val
            else:
                new_row_max = row_maxes[row_idx]  # Last element unchanged
            
            # New scope is max of all row maxes, with row_idx's max updated
            if new_row_max > row_maxes[row_idx]:
                # This row's max increased, new scope is max of old scope and new_row_max
                new_scope = max(current_scope, new_row_max)
            else:
                # This row's max decreased or same, new scope is max of other rows' maxes and new_row_max
                # We need to find the max of all rows except row_idx, then compare with new_row_max
                # For efficiency, if current_scope was from row_idx, we need to recompute
                # Simple approach: new_scope = max of all row_maxes with row_idx updated
                # But that's O(n_rows). Let's do it since n_rows is small (4)
                other_maxes = [row_maxes[i] for i in range(n_rows) if i != row_idx]
                new_scope = max(max(other_maxes) if other_maxes else 0, new_row_max)

            # Accept or reject with temperature-based criterion
            if new_scope < current_scope:
                # Always accept improvement
                rows[row_idx][pos] = new_val
                used_diffs.update(new_diffs)
                current_scope = new_scope
                row_maxes[row_idx] = new_row_max
                steps_since_improvement = 0

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]

            elif random.random() < pow(2.718281828, (current_scope - new_scope) / initial_temp):
                # Accept with probability based on temperature
                rows[row_idx][pos] = new_val
                used_diffs.update(new_diffs)
                current_scope = new_scope
                row_maxes[row_idx] = new_row_max
                steps_since_improvement += 1
            else:
                # Reject: restore old_diffs
                used_diffs.update(old_diffs)
                steps_since_improvement += 1

            # Cool down
            initial_temp *= cooling_rate

            # If stuck for too long, restart with a fresh randomization
            if steps_since_improvement > max_steps_without_improvement:
                # Randomly perturb a few positions to escape local minimum
                for _ in range(3):
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, k)
                    min_v = rows[r_idx][p_idx-1] + 1
                    max_v = rows[r_idx][p_idx+1] - 1 if p_idx < k else best_scope + 10
                    if min_v <= max_v:
                        new_v = random.randint(min_v, max_v)
                        old_v = rows[r_idx][p_idx]
                        if new_v != old_v:
                            # Remove old diffs
                            old_d = set()
                            for i in range(k + 1):
                                if i != p_idx:
                                    d = abs(old_v - rows[r_idx][i])
                                    old_d.add(d)
                            for d in old_d:
                                used_diffs.discard(d)
                            
                            # Check new diffs
                            new_d = set()
                            valid_perturb = True
                            for i in range(k + 1):
                                if i != p_idx:
                                    if i < p_idx:
                                        d = new_v - rows[r_idx][i]
                                    else:
                                        d = rows[r_idx][i] - new_v
                                    if d in used_diffs or d in new_d:
                                        valid_perturb = False
                                        break
                                    new_d.add(d)
                            
                            if valid_perturb:
                                rows[r_idx][p_idx] = new_v
                                used_diffs.update(new_d)
                                # Update row_maxes
                                if p_idx == k:
                                    row_maxes[r_idx] = new_v
                                current_scope = max(row_maxes)
                            else:
                                used_diffs.update(old_d)
                
                steps_since_improvement = 0
                initial_temp = max(1.0, current_scope / 10.0)

        return best_rows, best_scope

    # Step 1: Greedy construction with multiple restarts
    best_rows = None
    best_scope = float('inf')
    
    # Try greedy construction with different random seeds and scope limits
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Randomize the greedy construction slightly
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try to add next element with some randomization
                last_val = row[-1]
                max_val = 5000
                found = False
                # Try a random subset of candidate values
                candidates = list(range(last_val + 1, min(last_val + 200, max_val + 1)))
                random.shuffle(candidates)
                for x in candidates:
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    # Try full range
                    for x in range(last_val + 1, max_val + 1):
                        valid = True
                        for y in row:
                            d = x - y
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            row.append(x)
                            for y in row[:-1]:
                                used_diffs.add(x - y)
                            found = True
                            break
                if not found:
                    success = False
                    break
            if not success:
                break
            rows.append(row)
        
        if success and len(rows) == n_rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy
        rows = greedy_construct()
        if rows is None:
            rows = greedy_construct(scope_limit=5000)
        if rows is None:
            # Last resort
            used_diffs = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for y in row:
                        if x - y in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                rows.append(row)
            return rows
        best_rows = rows
        best_scope = max(max(r) for r in rows)

    # Step 2: Simulated annealing to improve
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_rows, best_scope = simulated_annealing(best_rows, remaining_time - 5)
        return best_rows

    return best_rows


def _validate(rows, n_rows, k):
    """Validate that the rows satisfy all constraints."""
    if len(rows) != n_rows:
        return False
    used_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, k + 1):
            if row[i] <= row[i-1]:
                return False
        # Check all differences are distinct
        for i in range(k + 1):
            for j in range(i+1, k + 1):
                d = row[j] - row[i]
                if d in used_diffs:
                    return False
                used_diffs.add(d)
    return True
