"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ri)
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_set(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Fast greedy baseline ----------
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

    # ---------- Polynomial spectral representation approach ----------
    # Key insight: if we represent each row as a sparse polynomial P_r(x) = sum x^{a_{r,i}},
    # the global difference constraint means that the autocorrelation of the combined
    # difference set must have no collisions. We use this to guide a more intelligent
    # search that prioritizes "spectral" gaps.
    
    # The product of all row polynomials' autocorrelations should be sparse.
    # We use this to bias our search toward rows that create "spectral gaps"
    # in the difference space, making it easier to pack remaining rows.
    
    def spectral_bias_row(row, used_diffs):
        """Compute a bias score based on how well this row's differences
        complement the used differences in a spectral sense."""
        row_diffs = row_diffs_set(row)
        # Count how many of this row's differences are in "gaps" of used_diffs
        # A gap is a difference value that is not used but whose neighbors are
        gap_count = 0
        for d in row_diffs:
            # Check if d is in a "gap" - i.e., d-1 or d+1 is in used_diffs but d is not
            if d not in used_diffs:
                if (d - 1 in used_diffs) or (d + 1 in used_diffs):
                    gap_count += 1
        return gap_count

    def build_row_with_spectral_bias(used_diffs, max_x, jitter=0.0):
        """Build a row with spectral bias - prefer differences that fill gaps."""
        row, x = [0], 0
        row_diffs = set()
        while len(row) < k + 1:
            best_candidate = None
            best_score = -1
            # Look ahead
            for trial_x in range(x + 1, min(x + 100, max_x + 1)):
                new = {trial_x - a for a in row}
                if (new & used_diffs) or (new & row_diffs):
                    continue
                # Compute spectral bias score
                score = spectral_bias_row(row + [trial_x], used_diffs)
                # Add a small random component for exploration
                score += random.random() * 0.5
                if score > best_score:
                    best_score = score
                    best_candidate = trial_x
            if best_candidate is None:
                return None
            # Apply jitter: sometimes skip the best candidate
            if jitter > 0 and random.random() < jitter and best_candidate < 200:
                # Pick a random valid candidate instead
                valid_candidates = []
                for trial_x in range(x + 1, min(x + 100, max_x + 1)):
                    new = {trial_x - a for a in row}
                    if not (new & used_diffs) and not (new & row_diffs):
                        valid_candidates.append(trial_x)
                if valid_candidates:
                    best_candidate = random.choice(valid_candidates)
            row.append(best_candidate)
            row_diffs |= {best_candidate - a for a in row[:-1]}
            x = best_candidate
        return row

    def local_polish_spectral(rows, max_passes=5):
        """Polish rows using spectral bias to reduce scope."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time_left() > 5 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Order rows by their max entry (largest first)
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 5:
                    break
                # Compute used differences from other rows
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                
                current_max = max(rows[i])
                # Try to build a better row with spectral bias
                for attempt in range(30):
                    if time_left() < 3:
                        break
                    jitter = 0.1 if attempt > 10 else 0.0
                    new_row = build_row_with_spectral_bias(used_other, current_max - 1, jitter)
                    if new_row is not None and max(new_row) < current_max:
                        # Verify no conflicts
                        new_diffs = row_diffs_set(new_row)
                        if not (new_diffs & used_other):
                            rows[i] = new_row
                            improved = True
                            break
        return rows

    def kick_spectral(rows):
        """Kick using spectral-aware perturbation."""
        rows = [list(r) for r in rows]
        # Target the most constrained rows
        weights = [max(r) + 1 for r in rows]
        total_weight = sum(weights)
        r = random.random() * total_weight
        cumsum = 0
        chosen_idx = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                chosen_idx = i
                break
        
        used_other = set()
        for j in range(n_rows):
            if j != chosen_idx:
                used_other |= row_diffs_set(rows[j])
        
        # Rebuild with high spectral bias and jitter
        for attempt in range(20):
            jitter = 0.3 + 0.02 * attempt
            new_row = build_row_with_spectral_bias(used_other, max(rows[chosen_idx]) + 50, jitter)
            if new_row is not None:
                rows[chosen_idx] = new_row
                break
        else:
            # Fallback to greedy
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
            rows[chosen_idx] = row
        
        return rows

    # Initial polish
    if time_left() > 15:
        polished = local_polish_spectral(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search with spectral bias
    no_improve_count = 0
    while time_left() > 25 and no_improve_count < 10:
        # 1. Local Search
        polished = local_polish_spectral(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 2. Kick
        if time_left() < 15:
            break
        kicked_rows = kick_spectral(best_rows)
        
        # 3. Local Search on kicked rows
        polished_kicked = local_polish_spectral(kicked_rows)
        
        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)
            no_improve_count = 0

    # Final polish
    if time_left() > 5:
        polished = local_polish_spectral(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
