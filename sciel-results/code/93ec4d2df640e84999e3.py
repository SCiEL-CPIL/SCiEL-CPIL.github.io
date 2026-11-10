import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using randomized greedy initialization
    with multiple restarts and targeted local search to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill

    def time_remaining():
        return deadline - time.time()

    def validate(rows):
        """Check if rows form a valid difference triangle set. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            if row:
                max_entry = max(max_entry, row[-1])
        return True, max_entry

    def compute_used_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_initial_randomized(n_rows, k, rng, max_x=None):
        """
        Randomized greedy initialization: for each row, try multiple candidate values
        and pick randomly among valid ones to explore different solution paths.
        """
        if max_x is None:
            max_x = 200
        
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if x > max_x:
                    return None
                
                # Collect all valid candidates in a window
                valid_candidates = []
                for candidate in range(x, min(x + 5, max_x + 1)):
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = candidate - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        valid_candidates.append((candidate, new_diffs))
                
                if not valid_candidates:
                    x += 1
                    continue
                
                # Pick randomly from valid candidates (bias toward smaller)
                # Weight: smaller candidates get higher weight
                weights = [1.0 / (1.0 + (c[0] - x) * 0.5) for c in valid_candidates]
                total_weight = sum(weights)
                r = rng.random() * total_weight
                cumulative = 0
                chosen_idx = 0
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_idx = i
                        break
                
                chosen_val, new_diffs = valid_candidates[chosen_idx]
                row.append(chosen_val)
                used_diffs.update(new_diffs)
                x = chosen_val + 1
                
                if time.time() > deadline - 10:
                    return None
            
            rows.append(row)
        
        return rows

    def try_move_element(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs) if successful, (False, None) otherwise.
        """
        row = rows[ri]
        current = row[ci]
        
        if new_val == current:
            return True, used_diffs
        
        # Collect old differences involving position ci
        old_diffs = set()
        for i in range(ci):
            old_diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - row[ci])
        
        # Remove old differences
        temp_used = used_diffs - old_diffs
        
        # Check if new_val is valid
        if ci > 0 and new_val <= row[ci - 1]:
            return False, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None
        
        new_diffs = set()
        valid = True
        for i in range(ci):
            d = new_val - row[i]
            if d <= 0 or d in temp_used:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            for j in range(ci + 1, len(row)):
                d = row[j] - new_val
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)
        
        if not valid:
            return False, None
        
        # Success: return the new used set
        new_used = temp_used | new_diffs
        return True, new_used

    def local_search_reduce_max(rows, n_rows, k, max_iters=5000):
        """
        Targeted local search: repeatedly try to reduce the maximum entry
        by moving elements in the row(s) with the largest values.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)
        current_score = best_score
        
        no_improve_count = 0
        max_no_improve = 200
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Find rows sorted by their max entry (descending)
            row_order = sorted(range(n_rows), key=lambda r: rows[r][-1], reverse=True)
            
            found_improvement = False
            
            for ri in row_order:
                if time.time() > deadline - 10:
                    break
                
                row = rows[ri]
                
                # Try to reduce each element from last to first (skip 0)
                for ci in range(len(row) - 1, 0, -1):
                    current = row[ci]
                    min_val = row[ci - 1] + 1
                    
                    # Try smaller values, starting from current-1
                    for new_val in range(current - 1, min_val - 1, -1):
                        success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                        if success:
                            rows[ri][ci] = new_val
                            used_diffs = new_used
                            new_score = max(r[-1] for r in rows)
                            if new_score < best_score:
                                best_score = new_score
                                current_score = new_score
                                no_improve_count = 0
                            found_improvement = True
                            break
                    
                    if found_improvement:
                        break
                
                if found_improvement:
                    break
            
            if not found_improvement:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        return rows, best_score

    def simulated_annealing(rows, n_rows, k, max_iters=10000, initial_temp=100.0):
        """
        Simulated annealing to minimize the maximum entry.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        current_score = max(row[-1] for row in rows)
        best_score = current_score
        best_rows = [row[:] for row in rows]
        
        rng = random.Random(int(time.time() * 1000) % 1000000)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            temp = initial_temp * (0.9998 ** iteration)
            if temp < 0.1:
                temp = 0.1
            
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.7:
                # Decrease an element
                ri = rng.randint(0, n_rows - 1)
                row = rows[ri]
                ci = rng.randint(1, len(row) - 1)
                current = row[ci]
                min_val = row[ci - 1] + 1
                max_val = (row[ci + 1] - 1) if ci < len(row) - 1 else current
                if min_val < max_val:
                    new_val = rng.randint(min_val, max_val)
                    if new_val < current:
                        success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                        if success:
                            new_score = max(r[-1] for r in rows)
                            delta = new_score - current_score
                            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(temp, 0.01))):
                                rows[ri][ci] = new_val
                                used_diffs = new_used
                                current_score = new_score
                                if current_score < best_score:
                                    best_score = current_score
                                    best_rows = [r[:] for r in rows]
                
            elif move_type < 0.9:
                # Increase an element (to escape local minima)
                ri = rng.randint(0, n_rows - 1)
                row = rows[ri]
                ci = rng.randint(1, len(row) - 1)
                current = row[ci]
                min_val = row[ci - 1] + 1
                max_val = (row[ci + 1] - 1) if ci < len(row) - 1 else current + 15
                if min_val < max_val:
                    new_val = rng.randint(min_val, max_val)
                    if new_val > current:
                        success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                        if success:
                            new_score = max(r[-1] for r in rows)
                            delta = new_score - current_score
                            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(temp, 0.01))):
                                rows[ri][ci] = new_val
                                used_diffs = new_used
                                current_score = new_score
                                if current_score < best_score:
                                    best_score = current_score
                                    best_rows = [r[:] for r in rows]
            
            else:
                # Swap two elements in different rows
                ri1 = rng.randint(0, n_rows - 1)
                ri2 = rng.randint(0, n_rows - 1)
                if ri1 != ri2:
                    row1 = rows[ri1]
                    row2 = rows[ri2]
                    ci1 = rng.randint(1, len(row1) - 1)
                    ci2 = rng.randint(1, len(row2) - 1)
                    
                    val1 = row1[ci1]
                    val2 = row2[ci2]
                    
                    # Try to swap values
                    old_diffs1 = set()
                    for i in range(ci1):
                        old_diffs1.add(row1[ci1] - row1[i])
                    for j in range(ci1 + 1, len(row1)):
                        old_diffs1.add(row1[j] - row1[ci1])
                    
                    old_diffs2 = set()
                    for i in range(ci2):
                        old_diffs2.add(row2[ci2] - row2[i])
                    for j in range(ci2 + 1, len(row2)):
                        old_diffs2.add(row2[j] - row2[ci2])
                    
                    temp_used = used_diffs - old_diffs1 - old_diffs2
                    
                    valid = True
                    new_diffs = set()
                    
                    if ci1 > 0 and val2 <= row1[ci1 - 1]:
                        valid = False
                    elif ci1 < len(row1) - 1 and val2 >= row1[ci1 + 1]:
                        valid = False
                    else:
                        for i in range(ci1):
                            d = val2 - row1[i]
                            if d <= 0 or d in temp_used:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            for j in range(ci1 + 1, len(row1)):
                                d = row1[j] - val2
                                if d <= 0 or d in temp_used:
                                    valid = False
                                    break
                                new_diffs.add(d)
                    
                    if valid:
                        if ci2 > 0 and val1 <= row2[ci2 - 1]:
                            valid = False
                        elif ci2 < len(row2) - 1 and val1 >= row2[ci2 + 1]:
                            valid = False
                        else:
                            for i in range(ci2):
                                d = val1 - row2[i]
                                if d <= 0 or d in (temp_used | new_diffs):
                                    valid = False
                                    break
                                new_diffs.add(d)
                            if valid:
                                for j in range(ci2 + 1, len(row2)):
                                    d = row2[j] - val1
                                    if d <= 0 or d in (temp_used | new_diffs):
                                        valid = False
                                        break
                                    new_diffs.add(d)
                    
                    if valid:
                        new_score = max(r[-1] for r in rows)
                        delta = new_score - current_score
                        if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(temp, 0.01))):
                            rows[ri1][ci1] = val2
                            rows[ri2][ci2] = val1
                            used_diffs = temp_used | new_diffs
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_rows = [r[:] for r in rows]
        
        return best_rows, best_score

    # Phase 1: Try multiple randomized greedy initializations
    best_rows = None
    best_score = float('inf')
    
    rng = random.Random(12345)
    
    for init_trial in range(20):
        if time_remaining() < 30:
            break
        
        # Vary the random seed and max_x for diversity
        trial_rng = random.Random(rng.randint(0, 1000000))
        max_x = 150 + init_trial * 10  # Gradually increase search space
        
        rows = greedy_initial_randomized(n_rows, k, trial_rng, max_x)
        if rows is None:
            continue
        
        valid, score = validate(rows)
        if not valid:
            continue
        
        if score < best_score:
            best_score = score
            best_rows = [row[:] for row in rows]
    
    # Fallback if no valid initialization found
    if best_rows is None:
        # Simple deterministic greedy
        used_diffs = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                else:
                    x += 1
            best_rows.append(row)
        best_score = validate(best_rows)[1]
    
    # Phase 2: Local search to reduce max
    improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=3000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows
    
    # Phase 3: Simulated annealing for further improvement
    improved_rows, improved_score = simulated_annealing(best_rows, n_rows, k, max_iters=15000, initial_temp=80.0)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows
    
    # Phase 4: More local search after SA
    improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=2000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows
    
    # Phase 5: Random restarts with perturbation
    for restart in range(20):
        if time_remaining() < 20:
            break
        
        # Perturb the solution
        perturbed = [row[:] for row in best_rows]
        rng = random.Random(1000 + restart * 7)
        for ri in range(n_rows):
            for ci in range(1, len(perturbed[ri])):
                if rng.random() < 0.2:
                    perturbed[ri][ci] += rng.randint(1, 15)
        
        # Validate perturbed
        valid, score = validate(perturbed)
        if not valid:
            continue
        
        # Local search from perturbed
        improved_rows, improved_score = local_search_reduce_max(perturbed, n_rows, k, max_iters=1500)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows
        
        # SA from perturbed
        improved_rows, improved_score = simulated_annealing(perturbed, n_rows, k, max_iters=5000, initial_temp=40.0)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows
    
    # Final local search
    improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=1000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows
    
    return best_rows
# EVOLVE-BLOCK-END
