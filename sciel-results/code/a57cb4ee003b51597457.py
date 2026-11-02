import time
import random
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid Simulated Annealing approach:
    1. Single-position moves for fine-tuning.
    2. Row-re-solve moves (randomized greedy reconstruction of a row) for escaping local minima.
    3. Incremental difference set updates for speed.
    4. Final right-to-left greedy polish.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    # ---- Helper Functions ----

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _global_max(rows):
        return max(max(row) for row in rows)

    def greedy_construct(n_rows, k):
        """Deterministic greedy initialization."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def randomized_greedy_row(forbidden_diffs, k, target_max, max_candidates=5, rng=None):
        """
        Build a single row greedily but with randomization among valid candidates.
        Tries to keep the max entry close to target_max.
        """
        if rng is None:
            rng = random
        row = [0]
        used_in_row = set()
        last_val = 0
        
        # We want to construct a row of length k+1 starting at 0.
        # We try to keep values <= target_max if possible, but allow slightly higher if stuck.
        # To be safe, we set a hard limit slightly above target_max.
        hard_limit = target_max + 50 
        
        for _ in range(k):
            candidates = []
            x = last_val + 1
            
            # Search for candidates
            # Optimization: Instead of scanning all x, we can jump, but for small k (6) and small scope (~100), scan is fine.
            while x < hard_limit:
                new_diffs = {x - a for a in row}
                # Check internal consistency
                if len(new_diffs) != len(row):
                    x += 1
                    continue
                # Check global consistency
                if new_diffs & forbidden_diffs:
                    x += 1
                    continue
                # Check local consistency (duplicates within this row's existing diffs)
                if new_diffs & used_in_row:
                    x += 1
                    continue
                
                candidates.append(x)
                if len(candidates) >= max_candidates:
                    break
                x += 1
            
            if not candidates:
                return None
            
            # Pick randomly from candidates.
            # Bias towards smaller values to keep scope low?
            # Uniform is often better for diversity in SA, but let's try a slight bias.
            # If we pick the smallest, it's deterministic greedy.
            # If we pick uniform, it's diverse.
            # Let's use a weighted choice: earlier candidates (smaller x) have higher weight.
            weights = [1.0 / (i + 1) for i in range(len(candidates))]
            total_weight = sum(weights)
            r = rng.random() * total_weight
            cumulative = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = i
                    break
            
            chosen_x = candidates[chosen_idx]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used_in_row |= new_diffs
            last_val = chosen_x
            
        return row

    def greedy_polish_right_to_left(rows, n_rows, k, deadline):
        """
        Right-to-left greedy polish: for each row, try to reduce each position
        from right to left to the smallest valid value.
        """
        best_rows = [r[:] for r in rows]
        used_diffs = set()
        row_diffs = []
        for r in best_rows:
            rd = _row_diffs(r)
            used_diffs |= rd
            row_diffs.append(rd)

        while time.time() < deadline - 1:
            progress = False
            row_order = list(range(n_rows))
            random.shuffle(row_order)

            for r_idx in row_order:
                if time.time() > deadline - 1:
                    break
                row = best_rows[r_idx]

                for pos in range(k, 0, -1):
                    old_val = row[pos]

                    # Diffs involving old_val at pos
                    diffs_involving_old = set()
                    for i in range(k + 1):
                        if i == pos:
                            continue
                        if i < pos:
                            diffs_involving_old.add(old_val - row[i])
                        else:
                            diffs_involving_old.add(row[i] - old_val)

                    forbidden_diffs = used_diffs - diffs_involving_old

                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1
                    if pos + 1 < k + 1:
                        upper_bound = min(upper_bound, row[pos + 1] - 1)

                    if upper_bound < lower_bound:
                        continue

                    found = None
                    for new_val in range(lower_bound, upper_bound + 1):
                        if new_val == old_val:
                            continue
                        valid = True
                        new_diffs = set()
                        for i in range(k + 1):
                            if i == pos:
                                continue
                            d = new_val - row[i] if i < pos else row[i] - new_val
                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)

                        if valid:
                            found = new_val
                            break

                    if found is not None and found < old_val:
                        # Update used_diffs
                        used_diffs -= diffs_involving_old
                        used_diffs |= new_diffs
                        row[pos] = found
                        # Update row_diffs for this row
                        row_diffs[r_idx] = _row_diffs(row)
                        progress = True

            if not progress:
                break

        return best_rows

    # ---- Main Logic ----

    # Initial Solution
    best_solution = greedy_construct(n_rows, k)
    best_scope = _global_max(best_solution)
    
    # State for SA
    current_rows = [r[:] for r in best_solution]
    current_row_diffs = [_row_diffs(r) for r in current_rows]
    current_all_diffs = set()
    for rd in current_row_diffs:
        current_all_diffs |= rd
    current_scope = best_scope

    T = 50.0
    T_min = 0.1
    cool = 0.99995
    no_improve = 0
    max_no_improve = 5000
    restarts = 0
    max_restarts = 10

    rng = random.Random(42)

    while time.time() < deadline - 5:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            # Restart from best
            current_rows = [r[:] for r in best_solution]
            current_row_diffs = [_row_diffs(r) for r in current_rows]
            current_all_diffs = set()
            for rd in current_row_diffs:
                current_all_diffs |= rd
            current_scope = best_scope
            T = 300.0
            no_improve = 0
            continue

        # Choose a move type
        # 70% single position, 30% row re-solve
        if rng.random() < 0.7:
            # --- Single Position Move ---
            # Pick row
            # Bias towards rows with high max entry
            row_maxes = [max(r) for r in current_rows]
            weights = [m ** 2 for m in row_maxes] # Heavier weight to larger rows
            total_w = sum(weights)
            r_sum = rng.random() * total_w
            cumulative = 0
            ri = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r_sum <= cumulative:
                    ri = i
                    break
            
            pi = rng.randint(1, k)
            row = current_rows[ri]
            old_val = row[pi]
            
            # Generate candidates
            min_allowed = row[pi - 1] + 1
            max_allowed = (row[pi + 1] - 1) if pi < k else current_scope + 20
            
            # Try nearby values
            candidates = []
            # Decreasing
            for d in range(1, min(20, old_val - min_allowed) + 1):
                candidates.append(old_val - d)
            # Increasing
            for d in range(1, min(10, max_allowed - old_val) + 1):
                candidates.append(old_val + d)
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue

            moved = False
            for new_val in candidates:
                if new_val < min_allowed or new_val > max_allowed:
                    continue
                if new_val == old_val:
                    continue
                
                # Check validity
                new_diffs = set()
                valid = True
                for i in range(k + 1):
                    if i == pi:
                        continue
                    d = new_val - row[i] if i < pi else row[i] - new_val
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check against global set (excluding this row's current diffs)
                # We need to check if new_diffs intersects with (all_diffs - current_row_diffs[ri])
                # Or more efficiently: 
                # 1. Remove old diffs involving pi from a temp set? No, just check intersection.
                # old_diffs_involving_pi = {old_val - row[i] if i < pi else row[i] - old_val for i in range(k+1) if i != pi}
                # forbidden = current_all_diffs - old_diffs_involving_pi
                # if new_diffs & forbidden: continue
                
                # Actually, simpler:
                # The set of diffs in the system is current_all_diffs.
                # If we change row[ri][pi], the diffs involving pi change.
                # Old diffs: D_old
                # New diffs: D_new
                # Other diffs: D_other = current_row_diffs[ri] - D_old
                # Global other diffs: G_other = current_all_diffs - current_row_diffs[ri]
                # We need D_new & D_other == empty AND D_new & G_other == empty.
                
                old_diffs_pi = set()
                for i in range(k + 1):
                    if i == pi: continue
                    old_diffs_pi.add(old_val - row[i] if i < pi else row[i] - old_val)
                
                other_row_diffs = current_row_diffs[ri] - old_diffs_pi
                global_other_diffs = current_all_diffs - current_row_diffs[ri]
                
                if new_diffs & other_row_diffs:
                    continue
                if new_diffs & global_other_diffs:
                    continue
                
                # Valid move
                # Calculate delta
                old_row_max = max(row)
                new_row = row[:]
                new_row[pi] = new_val
                new_row_max = max(new_row)
                
                # Update maxes
                row_maxes[ri] = new_row_max
                new_scope = max(row_maxes)
                row_maxes[ri] = old_row_max # revert for calculation
                
                delta = new_scope - current_scope
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if rng.random() < 0.5:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    # Apply move
                    # Update global diff set
                    current_all_diffs -= current_row_diffs[ri]
                    new_row_diffs = other_row_diffs | new_diffs
                    current_all_diffs |= new_row_diffs
                    
                    row[pi] = new_val
                    current_row_diffs[ri] = new_row_diffs
                    current_scope = new_scope
                    
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_solution = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
        else:
            # --- Row Re-solve Move ---
            # Pick a row to replace
            # Bias towards rows with high max entry or "random"
            row_maxes = [max(r) for r in current_rows]
            # Sort indices by max entry descending
            indices = sorted(range(n_rows), key=lambda i: row_maxes[i], reverse=True)
            # Pick from top 3 with probability, or random
            if rng.random() < 0.8:
                ri = indices[rng.randint(0, min(2, n_rows-1))]
            else:
                ri = rng.randint(0, n_rows - 1)
            
            # Remove this row's diffs from global set
            current_all_diffs -= current_row_diffs[ri]
            
            # Target max: current best scope (or current scope if we want to expand)
            # To find improvements, we try to build a row with max <= best_scope
            # If that fails, allow slightly higher.
            target_max = best_scope
            
            new_row = randomized_greedy_row(current_all_diffs, k, target_max, max_candidates=8, rng=rng)
            
            if new_row is None:
                # Try with a slightly larger target
                target_max = current_scope + 10
                new_row = randomized_greedy_row(current_all_diffs, k, target_max, max_candidates=8, rng=rng)
            
            if new_row is None:
                # Restore state
                current_all_diffs |= current_row_diffs[ri]
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            new_row_diffs = _row_diffs(new_row)
            
            # Check if this new row is valid with the rest (it should be by construction, 
            # but randomized_greedy_row uses current_all_diffs which has this row removed, 
            # so it's valid against others. Internal validity is guaranteed by construction).
            
            # Calculate delta
            old_row_max = max(current_rows[ri])
            new_row_max = max(new_row)
            
            row_maxes = [max(r) for r in current_rows]
            row_maxes[ri] = new_row_max
            new_scope = max(row_maxes)
            
            delta = new_scope - current_scope
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.5:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                # Apply move
                current_rows[ri] = new_row
                current_row_diffs[ri] = new_row_diffs
                current_all_diffs |= new_row_diffs
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_solution = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Reject: Restore state
                current_all_diffs |= current_row_diffs[ri]
                no_improve += 1
        
        T = max(T_min, T * cool)

    # Final Polish
    if time.time() < deadline - 2:
        polished = greedy_polish_right_to_left(best_solution, n_rows, k, deadline - 1)
        polished_scope = _global_max(polished)
        if polished_scope < best_scope:
            best_solution = polished
            best_scope = polished_scope

    return best_solution

# EVOLVE-BLOCK-END
