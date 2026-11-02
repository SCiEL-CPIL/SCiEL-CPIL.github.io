import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows and k differences (k+1 elements per row).
    Minimizes the largest entry (scope).
    
    Strategy: Simulated Annealing with row re-solve moves.
    - Start with a greedy solution.
    - Use SA to minimize the maximum entry.
    - Moves: Re-solve a specific row given the others, or swap rows.
    """
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit

    def get_row_diffs(row):
        """Get set of all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_solution(rows):
        """Validate that all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            # Check strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
            diffs = get_row_diffs(row)
            if len(diffs) != k * (k + 1) // 2:
                return False # Internal Sidon property violated
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True

    def greedy_row(used, k, limit=None, rng=None, randomize=False):
        """
        Build a single row greedily.
        If randomize is True and rng is provided, it picks from a window of valid candidates.
        """
        row = [0]
        x = 0
        if limit is None:
            limit = 100000
        while len(row) < k + 1:
            if rng is None or not randomize:
                # Deterministic smallest-first
                x += 1
                if x > limit:
                    return None
                new_diffs = [x - a for a in row]
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                if any(d in used for d in new_diffs):
                    continue
                row.append(x)
                for d in new_diffs:
                    used.add(d)
            else:
                # Randomized: find a few valid candidates and pick one
                candidates = []
                scan = x + 1
                # Look ahead a bit to find valid spots
                max_scan = min(x + 50, limit)
                while scan <= max_scan and len(candidates) < 5:
                    new_diffs = [scan - a for a in row]
                    if len(set(new_diffs)) == len(new_diffs):
                        if not any(d in used for d in new_diffs):
                            candidates.append(scan)
                    scan += 1
                
                if not candidates:
                    # Fallback to deterministic scan if no candidates in window
                    scan = x + 1
                    while scan <= limit:
                        new_diffs = [scan - a for a in row]
                        if len(set(new_diffs)) == len(new_diffs):
                            if not any(d in used for d in new_diffs):
                                candidates.append(scan)
                                break
                        scan += 1
                    
                if not candidates:
                    return None
                
                # Weighted choice: prefer smaller values
                # Weights decay exponentially
                weights = [1.0 / (i + 1) for i in range(len(candidates))]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0.0
                chosen = candidates[-1]
                for c, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        chosen = c
                        break
                
                row.append(chosen)
                for a in row[:-1]:
                    used.add(chosen - a)
                x = chosen
        return row

    def build_greedy_solution(n_rows, k, rng=None, randomize=False):
        """Build a full solution using greedy row construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, k, rng=rng, randomize=randomize)
            if row is None:
                # If greedy fails (shouldn't happen with high limit), restart or fail
                # For now, just return None to signal failure
                return None
            rows.append(row)
        return rows

    def resolve_row(ri, rows, k, rng=None, randomize=False, target_max=None):
        """
        Re-solve row ri, assuming other rows are fixed.
        Returns the new row or None.
        """
        # Collect differences from all other rows
        other_used = set()
        for i in range(n_rows):
            if i != ri:
                other_used |= get_row_diffs(rows[i])
        
        # Build new row avoiding other_used
        # We want to keep it small, so we use a limit
        # If target_max is provided, use it as a hard limit for the search
        # Otherwise, let it grow naturally but prefer small
        limit = target_max if target_max is not None else 20000
        new_row = greedy_row(other_used, k, limit=limit, rng=rng, randomize=randomize)
        return new_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    # 1. Initial Solution
    rng = random.Random(42)
    best_rows = build_greedy_solution(n_rows, k, rng=rng, randomize=False)
    if best_rows is None:
        # Fallback: extremely simple greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = [x - a for a in row]
                if len(set(new_diffs)) == len(new_diffs) and not any(d in used for d in new_diffs):
                    row.append(x)
                    for d in new_diffs:
                        used.add(d)
            best_rows.append(row)
    
    best_scope = get_scope(best_rows)
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # 2. Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    max_steps = 50000 # Upper bound, will be cut by time
    
    # We will run SA for a certain time, then restart if stuck
    restart_time = 10.0 # seconds per restart attempt
    
    while time.time() < deadline - 2.0:
        steps_in_this_run = 0
        restart_start = time.time()
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        T = 10.0
        
        while time.time() < deadline - 2.0 and (time.time() - restart_start) < restart_time:
            # Check time
            if time.time() > deadline - 2.0:
                break
            
            # Choose a move
            move_type = rng.random()
            if move_type < 0.8:
                # Re-solve a random row
                ri = rng.randint(0, n_rows - 1)
                # Try to improve: use a limit slightly smaller than current max if possible?
                # Or just re-solve with randomization to find a different valid row
                # To minimize scope, we should try to find a row with small max.
                # greedy_row with randomize=True tends to pick smaller values on average if window is tight.
                new_row = resolve_row(ri, current_rows, k, rng=rng, randomize=True)
                
                if new_row is not None:
                    # Check validity with respect to current_rows (excluding ri)
                    # resolve_row already ensures no conflict with others.
                    # But we must ensure new_row is internally valid (Sidon).
                    # greedy_row guarantees internal Sidon.
                    
                    # Calculate new scope
                    # Since only row ri changes, new_scope = max(max(new_row), max(others))
                    others_max = 0
                    for i in range(n_rows):
                        if i != ri:
                            others_max = max(others_max, max(current_rows[i]))
                    new_scope = max(max(new_row), others_max)
                    
                    delta = new_scope - current_scope
                    
                    # Acceptance
                    if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                        current_rows[ri] = new_row
                        current_scope = new_scope
                        
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in current_rows]
                            # Optional: restart SA with fresh T if we found a new best
                            # T = 10.0 
            else:
                # Swap two rows (doesn't change scope, but might help escape local minima in row assignment?)
                # Actually, swapping rows doesn't change the set of differences or the scope.
                # It only changes which row is which. Since rows are indistinguishable in the objective,
                # this move is useless for optimization unless we have row-specific constraints (we don't).
                # So we skip swaps to save time.
                pass
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
            steps_in_this_run += 1
            
            # Stagnation check: if no improvement in last 5000 steps, restart with perturbation
            if steps_in_this_run % 5000 == 0:
                 # Perturb: randomly re-solve 2 rows with higher randomization
                 for _ in range(2):
                     ri = rng.randint(0, n_rows - 1)
                     new_row = resolve_row(ri, current_rows, k, rng=rng, randomize=True)
                     if new_row is not None:
                         current_rows[ri] = new_row
                         # Recalculate scope
                         current_scope = get_scope(current_rows)
                         if current_scope < best_scope:
                             best_scope = current_scope
                             best_rows = [r[:] for r in current_rows]
                 T = 5.0 # Reset temperature
        
        # End of restart block
        # If we used up the restart time, loop back to start a new SA phase with fresh best_rows

    # Final validation
    if not is_valid_solution(best_rows):
        # This should not happen, but fallback to greedy
        return build_greedy_solution(n_rows, k, rng=random.Random(1), randomize=False)
        
    return best_rows

# EVOLVE-BLOCK-END
