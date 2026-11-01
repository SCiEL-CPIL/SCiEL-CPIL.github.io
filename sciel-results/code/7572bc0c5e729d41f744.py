"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_list(row):
        """All positive differences in a row (as a list)."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_list(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Per-row local optimization (refinement machinery) ----------
    def row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def try_reduce_row(rows, row_idx, used_other, max_attempts=100):
        """Try to find a row at row_idx with smaller max entry (same k+1 size)."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 2:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            # Randomized greedy for this row to escape local minima in row construction
            jitter = 0.1 if attempt > 10 else 0.0
            while len(row) < k + 1:
                found = False
                # Look ahead a bit
                for trial_x in range(x + 1, x + 50):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        # Apply jitter: sometimes skip a valid candidate to explore
                        if jitter > 0 and random.random() < jitter and trial_x < 200:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)
        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        # Limit the number of full passes to avoid spending too much time here
        # if we are in an ILS loop
        max_passes = 5 
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row(rows, i, used_other, max_attempts=50)
                if new_max < max(rows[i]):
                    old_scope = scope(rows)
                    rows[i] = new_row
                    new_scope = scope(rows)
                    if new_scope < old_scope:
                        improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        # Strategy: Rebuild 1 or 2 random rows with high randomness
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)
        
        for idx in indices:
            # Calculate used diffs from other rows
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            # Rebuild the row with high jitter
            row, x = [0], 0
            current_row_diffs = set()
            success = False
            # Try a few times to find a valid row, even if larger
            for attempt in range(20):
                row, x = [0], 0
                current_row_diffs = set()
                valid_row = True
                while len(row) < k + 1:
                    found = False
                    # Wider search range for kicks
                    for trial_x in range(x + 1, x + 100):
                        new = {trial_x - a for a in row}
                        if not (new & used_other) and not (new & current_row_diffs):
                            # High jitter: skip valid candidates often
                            if random.random() < 0.4 and trial_x < 300:
                                continue
                            row.append(trial_x)
                            current_row_diffs |= new
                            found = True
                            break
                    if not found:
                        valid_row = False
                        break
                    x = row[-1]
                
                if valid_row:
                    # Check if this new row is valid with the rest (it should be by construction)
                    # But we must ensure it doesn't conflict with itself (handled) and others (handled)
                    rows[idx] = row
                    success = True
                    break
            
            if not success:
                # If we failed to rebuild, keep the original row
                # This might make the solution invalid temporarily, but local_polish will fix it
                # Or we revert to a greedy row
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # ---------- Simulated Annealing with spectral shift moves ----------
    def energy(rows, t):
        """Energy = scope + penalty for collisions (smooth surrogate)."""
        all_diffs = []
        for row in rows:
            for d in row_diffs_list(row):
                all_diffs.append(d)
        
        # Count collisions
        from collections import Counter
        counts = Counter(all_diffs)
        collisions = sum(c * (c - 1) // 2 for c in counts.values())
        
        # Scope term
        sc = max(max(row) for row in rows)
        
        # Smooth penalty: use log-sum-exp style for max, but for discrete we use quadratic
        # Energy = sc + lambda * collisions
        lam = 1.0
        return sc + lam * collisions

    def spectral_shift(rows):
        """Apply random phase rotations to row elements (shifts)."""
        new_rows = [list(r) for r in rows]
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        # Shift some elements by small random amounts
        for _ in range(random.randint(1, 3)):
            # Pick a random position in the row (not 0)
            pos = random.randint(1, k)
            # Shift by -2, -1, 1, or 2
            delta = random.choice([-2, -1, 1, 2])
            old_val = new_rows[r_idx][pos]
            new_val = old_val + delta
            
            # Check if shift keeps row strictly increasing and starts at 0
            if new_val <= 0:
                continue
            if pos < k and new_val >= new_rows[r_idx][pos + 1]:
                continue
            if pos > 0 and new_val <= new_rows[r_idx][pos - 1]:
                continue
            
            new_rows[r_idx][pos] = new_val
        return new_rows

    def simulated_annealing(rows, time_budget):
        """SA loop with spectral shift moves."""
        current_rows = [list(r) for r in rows]
        current_energy = energy(current_rows, 0)
        best_rows = [list(r) for r in current_rows]
        best_energy = current_energy
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time and T > T_min:
            # Generate neighbor via spectral shift
            neighbor = spectral_shift(current_rows)
            
            # Check validity of neighbor (must be valid for energy to be meaningful)
            if not is_valid(neighbor):
                # Invalid neighbor: reject
                T *= cooling
                continue
            
            neighbor_energy = energy(neighbor, 0)
            
            # Acceptance criterion
            if neighbor_energy < current_energy:
                current_rows = neighbor
                current_energy = neighbor_energy
                if current_energy < best_energy:
                    best_rows = [list(r) for r in current_rows]
                    best_energy = current_energy
            else:
                delta_e = neighbor_energy - current_energy
                if random.random() < math.exp(-delta_e / T):
                    current_rows = neighbor
                    current_energy = neighbor_energy
            
            T *= cooling
        
        return best_rows

    # Initial polish
    if time_left() > 10:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    # We use a loop that alternates between Local Search and Kicks
    # This is structurally different from a single long SA run.
    while time_left() > 20:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
        
        # 2. Kick
        if time_left() < 15:
            break
            
        kicked_rows = kick(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)
        
        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
        else:
            # If the kick didn't lead to improvement, we might stick to the previous best
            # and try a different kick next iteration.
            # This is the standard ILS behavior.
            pass

    # 5. Simulated Annealing phase (new)
    if time_left() > 10:
        sa_time = min(time_left() - 5, 30)  # Use up to 30s for SA
        if sa_time > 5:
            sa_result = simulated_annealing(best_rows, sa_time)
            if scope(sa_result) < best_scope:
                best_rows = sa_result
                best_scope = scope(best_rows)
                # Polish the SA result
                if time_left() > 5:
                    polished_sa = local_polish(best_rows)
                    if scope(polished_sa) < best_scope:
                        best_rows = polished_sa
                        best_scope = scope(polished_sa)

    # Final polish to ensure we squeeze out any last drops
    if time_left() > 5:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
