"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard: kill before 150s wall-clock limit
    # Use 145s to be safe
    deadline = time.time() + 145.0
    
    n_marks = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Helper: compute all positive differences for a list of rows
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

    # --- Cost Function ---
    # Returns (max_entry, conflicts)
    # We use a combined cost: max_entry * 1000 + conflicts
    # This prioritizes reducing max_entry, but heavily penalizes conflicts
    def compute_cost(rows):
        used = set()
        max_val = 0
        conflicts = 0
        for r in range(n_rows):
            row = rows[r]
            # Update max
            if row[-1] > max_val:
                max_val = row[-1]
            # Check differences
            for i in range(n_marks):
                for j in range(i + 1, n_marks):
                    d = row[j] - row[i]
                    if d <= 0:
                        # Invalid structure, huge penalty
                        return 10**9, 10**9
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
        return max_val, conflicts

    def get_score(rows):
        m, c = compute_cost(rows)
        return m * 1000 + c

    # --- Local Search: Try to fix conflicts and reduce max ---
    def local_search(rows, time_limit):
        current_rows = [r[:] for r in rows]
        current_max, current_conf = compute_cost(current_rows)
        current_score = current_max * 1000 + current_conf
        best_rows = [r[:] for r in current_rows]
        best_score = current_score
        best_max = current_max
        best_conf = current_conf

        no_improve = 0
        max_no_improve = 1000
        start_time = time.time()

        while time.time() - start_time < time_limit and no_improve < max_no_improve:
            improved = False
            # Try to reduce max entry by shifting the row with the largest entry
            if best_conf == 0:
                # If valid, try to shrink
                max_r = max(range(n_rows), key=lambda r: current_rows[r][-1])
                # Try shifting the last element down by 1
                old_val = current_rows[max_r][-1]
                if old_val > current_rows[max_r][-2] + 1:
                    current_rows[max_r][-1] = old_val - 1
                    new_max, new_conf = compute_cost(current_rows)
                    new_score = new_max * 1000 + new_conf
                    if new_score < current_score:
                        current_score = new_score
                        current_max = new_max
                        current_conf = new_conf
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_max = new_max
                            best_conf = new_conf
                            best_rows = [r[:] for r in current_rows]
                    else:
                        current_rows[max_r][-1] = old_val
            
            # Try to fix conflicts or reduce max by adjusting individual elements
            # We iterate through rows and positions
            for r in range(n_rows):
                for pos in range(1, k + 1):
                    # Determine valid range for this position
                    lower = current_rows[r][pos - 1] + 1
                    upper = current_rows[r][pos + 1] - 1 if pos < k else current_rows[r][k] + 10
                    
                    if lower > upper:
                        continue
                    
                    old_val = current_rows[r][pos]
                    # Try small perturbations
                    candidates = set()
                    # Try to shrink if possible
                    if old_val > lower:
                        candidates.add(old_val - 1)
                        candidates.add(old_val - 2)
                    # Try random nearby
                    for _ in range(3):
                        candidates.add(random.randint(lower, upper))
                    
                    for val in candidates:
                        if val == old_val:
                            continue
                        if val < lower or val > upper:
                            continue
                        
                        current_rows[r][pos] = val
                        new_max, new_conf = compute_cost(current_rows)
                        new_score = new_max * 1000 + new_conf
                        
                        if new_score < current_score:
                            current_score = new_score
                            current_max = new_max
                            current_conf = new_conf
                            improved = True
                            if new_score < best_score:
                                best_score = new_score
                                best_max = new_max
                                best_conf = new_conf
                                best_rows = [r[:] for r in current_rows]
                            break # Move to next position/row if improved
                        else:
                            current_rows[r][pos] = old_val
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
        
        return best_rows, best_score, best_conf

    # --- Simulated Annealing ---
    def simulated_annealing(rows, time_limit):
        current_rows = [r[:] for r in rows]
        current_max, current_conf = compute_cost(current_rows)
        current_score = current_max * 1000 + current_conf
        
        best_rows = [r[:] for r in current_rows]
        best_score = current_score
        best_max = current_max
        best_conf = current_conf

        T = 100.0
        T_min = 0.01
        cooling = 0.9999
        start_time = time.time()

        while time.time() - start_time < time_limit and T > T_min:
            # Pick a random move
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            lower = current_rows[r][pos - 1] + 1
            upper = current_rows[r][pos + 1] - 1 if pos < k else current_rows[r][k] + 50
            
            if lower > upper:
                continue
            
            old_val = current_rows[r][pos]
            
            # Choose new value
            # Bias towards smaller values to help reduce max
            if random.random() < 0.5:
                # Small perturbation
                delta = random.randint(-3, 3)
                new_val = old_val + delta
            else:
                # Random in range, biased towards lower end
                span = upper - lower
                if span > 0:
                    # Use a triangular-like distribution favoring lower values
                    new_val = lower + int(random.random() ** 2 * span)
                else:
                    new_val = lower

            if new_val < lower or new_val > upper:
                continue
            if new_val == old_val:
                continue

            # Apply move
            current_rows[r][pos] = new_val
            
            # Check validity of row structure (strictly increasing)
            # Since we picked from [lower, upper], it should be valid, but let's be safe
            row = current_rows[r]
            valid_struct = all(row[i] < row[i+1] for i in range(n_marks-1))
            
            if not valid_struct:
                current_rows[r][pos] = old_val
                continue

            new_max, new_conf = compute_cost(current_rows)
            new_score = new_max * 1000 + new_conf

            delta_score = new_score - current_score
            
            # Acceptance criterion
            if delta_score <= 0 or random.random() < math.exp(-delta_score / T):
                current_score = new_score
                current_max = new_max
                current_conf = new_conf
                
                # Update best if valid and better
                if new_conf == 0 and new_max < best_max:
                    best_max = new_max
                    best_score = new_score
                    best_rows = [r[:] for r in current_rows]
                elif new_score < best_score and new_conf == 0:
                    # This case is covered above if new_max < best_max
                    # But if new_max == best_max and score is same, no change
                    pass
            else:
                # Reject
                current_rows[r][pos] = old_val

            T *= cooling

        return best_rows, best_score, best_conf

    # --- Main Logic ---
    # 1. Greedy Init
    best_rows = greedy_init()
    best_max, best_conf = compute_cost(best_rows)
    
    # 2. Local Search to improve
    ls_rows, ls_score, ls_conf = local_search(best_rows, time_limit=5.0)
    if ls_conf == 0 and ls_score < best_max * 1000 + best_conf:
        best_rows = ls_rows
        best_max, best_conf = compute_cost(best_rows)

    # 3. Simulated Annealing for the rest of the time
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        sa_rows, sa_score, sa_conf = simulated_annealing(best_rows, time_limit=remaining_time - 0.5)
        if sa_conf == 0 and sa_score < best_max * 1000 + best_conf:
            best_rows = sa_rows
            best_max, best_conf = compute_cost(best_rows)
        elif sa_conf == 0:
            # If SA found a valid solution with same or worse score, keep best
            # But if SA found valid and current best was invalid (shouldn't happen), take SA
            if best_conf > 0:
                best_rows = sa_rows
                best_max, best_conf = compute_cost(best_rows)

    # 4. Final Local Search if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 1.0 and best_conf > 0:
        # Try to fix any remaining conflicts with LS
        ls_rows, ls_score, ls_conf = local_search(best_rows, time_limit=remaining_time - 0.5)
        if ls_conf < best_conf or (ls_conf == best_conf and ls_score < best_max * 1000 + best_conf):
            best_rows = ls_rows
            best_max, best_conf = compute_cost(best_rows)

    # Return the best valid rows found
    # If no valid rows found (best_conf > 0), we must return something.
    # The task requires valid rows. If we can't find a valid one, we return the best invalid one?
    # No, "Valid when all positive differences... are distinct".
    # If we return invalid, score is 0.
    # So we must ensure we return a valid solution if one exists.
    # The greedy init is always valid. So best_rows from greedy is valid.
    # We only update best_rows if the new candidate is valid (conf==0) and better.
    # So best_rows should always be valid.
    
    # Double check validity before returning
    if not is_valid(best_rows):
        # Fallback to greedy init which is guaranteed valid
        best_rows = greedy_init()

    return best_rows
# EVOLVE-BLOCK-END
