import time
import random
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a combination of greedy construction, right-to-left greedy polishing,
    and targeted simulated annealing focusing on the global maximum.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def greedy_construct(n_rows, k, seed=None):
        """
        Greedy construction: build rows one by one, filling with smallest valid integers.
        """
        if seed is not None:
            random.seed(seed)
        
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            used_local = set()
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    # Check internal consistency and global uniqueness
                    if not (new_diffs & used_local) and not (new_diffs & used_diffs):
                        row.append(x)
                        used_local |= new_diffs
                        break
                    x += 1
            used_diffs |= used_local
            rows.append(row)
        
        # Shuffle rows to avoid bias in later optimizations
        random.shuffle(rows)
        return rows

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _global_max(rows):
        return max(max(row) for row in rows)

    def greedy_polish_right_to_left(rows, n_rows, k, deadline):
        """
        Polishes the solution by trying to reduce each element from right to left
        to the smallest valid value given the current set of used differences.
        This is a very strong heuristic for DTS.
        """
        best_rows = [r[:] for r in rows]
        row_diffs = [_row_diffs(r) for r in best_rows]
        
        # Compute global used diffs
        all_used = set()
        for rd in row_diffs:
            all_used |= rd

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Iterate rows in a fixed order (or random)
            for ri in range(n_rows):
                row = best_rows[ri]
                rd = row_diffs[ri]
                
                # For each position from right to left (excluding 0)
                for pi in range(k, 0, -1):
                    old_val = row[pi]
                    # Diffs involving old_val
                    old_diffs_involving = set()
                    for i in range(k + 1):
                        if i == pi:
                            continue
                        if i < pi:
                            old_diffs_involving.add(old_val - row[i])
                        else:
                            old_diffs_involving.add(row[i] - old_val)
                    
                    # Diffs that must remain unique (not in this row's other diffs, not in other rows)
                    # Forbidden set: all_used - old_diffs_involving
                    # But we also need to ensure new diffs don't conflict with row's other diffs (not involving pi)
                    row_other_diffs = rd - old_diffs_involving
                    forbidden = all_used - old_diffs_involving | row_other_diffs 
                    # Actually, simpler: 
                    # New diffs must not be in (all_used - old_diffs_involving)
                    # AND New diffs must not be in (row_other_diffs)
                    # So Forbidden = (all_used - old_diffs_involving) | row_other_diffs
                    # Note: row_other_diffs is a subset of all_used, so Forbidden = all_used - old_diffs_involving
                    # Wait, if a diff is in row_other_diffs, it is in all_used. 
                    # If we remove old_diffs_involving from all_used, row_other_diffs remains.
                    # So Forbidden = all_used - old_diffs_involving is correct.
                    
                    forbidden = all_used - old_diffs_involving

                    lower_bound = row[pi - 1] + 1
                    upper_bound = old_val - 1
                    if pi < k:
                        upper_bound = min(upper_bound, row[pi + 1] - 1)
                    
                    if upper_bound < lower_bound:
                        continue

                    found_val = None
                    # Try to find the smallest valid value
                    for new_val in range(lower_bound, upper_bound + 1):
                        if new_val == old_val:
                            continue
                        
                        new_diffs_involving = set()
                        valid = True
                        for i in range(k + 1):
                            if i == pi:
                                continue
                            if i < pi:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val
                            
                            # Check against forbidden
                            if d in forbidden:
                                valid = False
                                break
                            # Check against other new diffs in this move
                            if d in new_diffs_involving:
                                valid = False
                                break
                            new_diffs_involving.add(d)
                        
                        if valid:
                            found_val = new_val
                            break
                    
                    if found_val is not None and found_val < old_val:
                        # Update structures
                        all_used -= old_diffs_involving
                        all_used |= new_diffs_involving
                        
                        row[pi] = found_val
                        row_diffs[ri] = (row_diffs[ri] - old_diffs_involving) | new_diffs_involving
                        improved = True
                        # If the global max decreased, we might want to restart the loop
                        # But continuing to polish other rows is also beneficial.
                        
        return best_rows

    def simulated_annealing(rows, n_rows, k, deadline):
        """
        Targeted SA: Focus on reducing the maximum value of the rows that define the global max.
        """
        current_rows = [r[:] for r in rows]
        row_diffs = [_row_diffs(r) for r in current_rows]
        all_used = set()
        for rd in row_diffs:
            all_used |= rd
        
        current_max = _global_max(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_max = current_max
        
        T = 100.0
        T_min = 0.1
        cool = 0.9995
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Identify rows that have the current maximum
            max_rows = [i for i in range(n_rows) if max(current_rows[i]) == current_max]
            
            if not max_rows:
                break
                
            # Pick a row from the max rows
            ri = random.choice(max_rows)
            row = current_rows[ri]
            rd = row_diffs[ri]
            
            # Pick a position to move. Bias towards larger positions (right side)
            # because they are more likely to be the maximum or close to it.
            pi = random.randint(1, k)
            
            old_val = row[pi]
            
            # Generate candidates
            # Try to decrease the value significantly
            candidates = []
            min_allowed = row[pi - 1] + 1
            max_allowed = row[pi + 1] - 1 if pi < k else current_max + 10
            
            # Decreasing moves
            for d in range(1, min(50, old_val - min_allowed + 1)):
                candidates.append(old_val - d)
            
            # Increasing moves (less likely, but needed for escaping local minima)
            for d in range(1, min(20, max_allowed - old_val + 1)):
                candidates.append(old_val + d)
                
            # Random jump
            if random.random() < 0.2:
                candidates.append(random.randint(min_allowed, max_allowed))
            
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                if new_val < min_allowed or new_val > max_allowed:
                    continue
                if new_val == old_val:
                    continue
                
                # Compute new diffs involving pi
                new_diffs_involving = set()
                valid = True
                for i in range(k + 1):
                    if i == pi:
                        continue
                    if i < pi:
                        d = new_val - row[i]
                    else:
                        d = row[i] - new_val
                    
                    if d in new_diffs_involving:
                        valid = False
                        break
                    new_diffs_involving.add(d)
                
                if not valid:
                    continue
                
                # Check against global used diffs
                # Diffs that will be removed: old_diffs_involving
                old_diffs_involving = set()
                for i in range(k + 1):
                    if i == pi:
                        continue
                    if i < pi:
                        old_diffs_involving.add(old_val - row[i])
                    else:
                        old_diffs_involving.add(row[i] - old_val)
                
                # Forbidden: all_used - old_diffs_involving
                forbidden = all_used - old_diffs_involving
                
                for d in new_diffs_involving:
                    if d in forbidden:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate new max
                # Only row ri changes
                new_row_max = max(row[:pi] + [new_val] + row[pi+1:])
                other_maxes = [max(current_rows[j]) for j in range(n_rows) if j != ri]
                new_global_max = max(new_row_max, max(other_maxes) if other_maxes else 0)
                
                delta = new_global_max - current_max
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.5:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Update state
                    all_used -= old_diffs_involving
                    all_used |= new_diffs_involving
                    
                    row[pi] = new_val
                    row_diffs[ri] = (row_diffs[ri] - old_diffs_involving) | new_diffs_involving
                    current_max = new_global_max
                    
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
                
            T = max(T_min, T * cool)
            
            # Restart if stuck
            if no_improve > max_no_improve:
                current_rows = [r[:] for r in best_rows]
                row_diffs = [_row_diffs(r) for r in current_rows]
                all_used = set()
                for rd in row_diffs:
                    all_used |= rd
                current_max = best_max
                no_improve = 0
                T = 100.0

        return best_rows, best_max

    # Main execution
    best_solution = None
    best_scope = float('inf')

    # Run multiple trials
    for trial in range(10):
        if time.time() > deadline - 10:
            break
            
        seed = trial * 1000 + 42
        init_rows = greedy_construct(n_rows, k, seed=seed)
        
        # Polish
        polished_rows = greedy_polish_right_to_left(init_rows, n_rows, k, deadline - 5)
        scope = _global_max(polished_rows)
        
        if scope < best_scope:
            best_scope = scope
            best_solution = polished_rows
            
        # SA refinement on the polished solution
        if time.time() < deadline - 5:
            sa_rows, sa_scope = simulated_annealing(polished_rows, n_rows, k, deadline - 2)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_solution = sa_rows

    # Final polish on the best solution found
    if time.time() < deadline - 1 and best_solution:
        final_rows = greedy_polish_right_to_left(best_solution, n_rows, k, deadline - 0.5)
        final_scope = _global_max(final_rows)
        if final_scope < best_scope:
            best_scope = final_scope
            best_solution = final_rows

    # Fallback if something went wrong (should not happen)
    if best_solution is None:
        best_solution = greedy_construct(n_rows, k)

    return best_solution

# EVOLVE-BLOCK-END
