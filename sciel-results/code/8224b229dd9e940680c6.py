import time
import random
import math
import sys

# Ensure we have enough recursion depth if needed, though iterative loops are used
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: greedy initialization, then Iterated Local Search
    with Simulated Annealing to minimize the largest entry.
    """
    start_time = time.time()
    # Safety margin: 150s limit, use 145s
    deadline = start_time + 145.0 

    # Lower bound for scope (not strictly used for logic, but good to know)
    # total_diffs = n_rows * k * (k + 1) // 2
    
    # 1. Greedy Initialization
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            # We need k more elements
            while len(row) < k + 1:
                found = False
                # Search for the smallest valid next element
                # Start from current_max + 1
                # Upper bound: heuristic, maybe current_max + 500
                for x in range(current_max + 1, current_max + 1000):
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        current_max = x
                        found = True
                        break
                if not found:
                    # If we can't find it in this range, try wider or restart
                    # For robustness, return None to signal failure, 
                    # but greedy is usually very fast for small k
                    return None
            rows.append(row)
        return rows, used_diffs

    init_result = greedy_construct()
    if not init_result:
        # Fallback: very sparse random construction (unlikely to be needed for 7,6)
        # Just return a valid but potentially large set
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                while True:
                    x = random.randint(row[-1] + 1, row[-1] + 50)
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        break
            rows.append(row)
        return rows

    best_solution, best_used_diffs = init_result
    best_score = max(max(r) for r in best_solution)
    
    # Helper to compute row diffs
    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    # Initialize state for the first solution
    current_rows = [r[:] for r in best_solution]
    current_used_diffs = set(best_used_diffs)
    current_row_diffs = [get_row_diffs(r) for r in current_rows]
    current_row_maxes = [max(r) for r in current_rows]
    current_score = max(current_row_maxes)

    # Simulated Annealing Parameters
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    def sa_optimize(rows, used_diffs, row_diffs, row_maxes, current_score, max_time=5.0):
        """
        Runs Simulated Annealing to minimize the max entry.
        Returns updated state.
        """
        start_sa = time.time()
        # We will run for a fixed number of steps or until time limit
        steps = 0
        max_steps = 50000 # Safety cap
        
        while time.time() - start_sa < max_time and steps < max_steps:
            steps += 1
            
            # Choose a row: bias towards rows with max value
            if random.random() < 0.7:
                max_val = max(row_maxes)
                max_indices = [i for i, m in enumerate(row_maxes) if m == max_val]
                r_idx = random.choice(max_indices)
            else:
                r_idx = random.randint(0, n_rows - 1)
            
            # Choose a position: 1..k (0 is fixed)
            pos = random.randint(1, k)
            
            old_val = rows[r_idx][pos]
            lower_bound = rows[r_idx][pos - 1] + 1
            upper_bound = rows[r_idx][pos + 1] - 1 if pos < k else current_score + 50
            
            if upper_bound < lower_bound:
                continue
            
            # Generate candidate value
            # Bias towards decreasing values (lowering scope)
            # but allow increases to escape local minima
            if random.random() < 0.6:
                # Try to decrease
                dec_range = old_val - lower_bound
                if dec_range > 0:
                    # Random step in [1, dec_range]
                    step = random.randint(1, min(dec_range, 20)) 
                    new_val = old_val - step
                else:
                    new_val = old_val # No change possible, skip
            else:
                # Try to increase or random jump
                inc_range = upper_bound - old_val
                if inc_range > 0:
                    step = random.randint(1, min(inc_range, 20))
                    new_val = old_val + step
                else:
                    new_val = old_val
            
            if new_val == old_val:
                continue
            
            # Validate new value
            # 1. Check internal row consistency (strictly increasing is guaranteed by bounds, 
            #    but we must check difference uniqueness within the row)
            # 2. Check global difference uniqueness
            
            # Calculate new diffs for this row
            new_row = rows[r_idx][:]
            new_row[pos] = new_val
            
            new_diffs = set()
            n = len(new_row)
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = new_row[j] - new_row[i]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check global conflicts
            # The diffs that are currently "used" by this row are row_diffs[r_idx]
            # The new diffs are new_diffs
            # Conflicts exist if (new_diffs - row_diffs[r_idx]) intersects with 
            # (used_diffs - row_diffs[r_idx])
            
            # Efficient check:
            # Any diff in new_diffs that is NOT in old row_diffs must not be in global used_diffs
            # OR if it is in global used_diffs, it must be in old row_diffs (which means it's being "moved")
            
            # Actually, simpler:
            # Forbidden = used_diffs - row_diffs[r_idx]
            # If new_diffs intersects Forbidden, it's invalid.
            
            forbidden = used_diffs - row_diffs[r_idx]
            if new_diffs & forbidden:
                continue
            
            # Valid move!
            # Calculate delta in score
            old_row_max = row_maxes[r_idx]
            new_row_max = max(new_row)
            
            # Update global max
            # We need to know if the global max changes
            # current_score is max(row_maxes)
            
            # If old_row_max was the unique max, and new_row_max < old_row_max, score drops.
            # If old_row_max was max, and new_row_max > old_row_max, score rises.
            # If old_row_max < current_score, score is unchanged unless new_row_max > current_score.
            
            # Let's compute new_score accurately
            # Temporarily update row_maxes
            old_max_list = row_maxes[:]
            row_maxes[r_idx] = new_row_max
            new_score = max(row_maxes)
            delta = new_score - current_score
            
            # Acceptance Criterion
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                # Accept neutral moves with some probability to shuffle
                if random.random() < 0.3:
                    accept = True
            else:
                # Accept worse moves based on temperature
                if T > T_min:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Commit the move
                # Update used_diffs
                used_diffs.difference_update(row_diffs[r_idx])
                used_diffs.update(new_diffs)
                
                rows[r_idx] = new_row
                row_diffs[r_idx] = new_diffs
                row_maxes[r_idx] = new_row_max
                current_score = new_score
                
                # Cool down
                T = max(T_min, T * cooling_rate)
                
            else:
                # Revert row_maxes if we modified it for calculation
                row_maxes[r_idx] = old_max_list[r_idx]
                # No need to revert used_diffs/rows as we didn't commit
                # But we did calculate new_score, which is fine.
                # We should still cool down? Usually yes, to ensure progress in time.
                T = max(T_min, T * cooling_rate)

        return rows, used_diffs, row_diffs, row_maxes, current_score

    # Iterated Local Search Loop
    # We will alternate between SA optimization and perturbation
    while time.time() < deadline - 10:
        
        # 1. Optimize current state using SA
        # Give SA a chunk of time, e.g., 2-5 seconds
        sa_time_budget = min(5.0, deadline - time.time() - 10)
        if sa_time_budget <= 1:
            break
            
        current_rows, current_used_diffs, current_row_diffs, current_row_maxes, current_score = \
            sa_optimize(current_rows, current_used_diffs, current_row_diffs, current_row_maxes, current_score, sa_time_budget)
        
        # 2. Check for improvement
        if current_score < best_score:
            best_score = current_score
            best_solution = [r[:] for r in current_rows]
            best_used_diffs = set(current_used_diffs)
            # Reset temperature for next SA run? Or keep it low?
            # Usually, after a good move, we want to explore locally again.
            # Let's reset T to a moderate value to allow some exploration.
            T = 50.0 
            
        # 3. Perturbation
        # If we are stuck (or just to escape local minima), perturb the best solution
        # We perturb the current state (which is either best or a neighbor)
        # Let's perturb the current state.
        
        # Number of random moves
        n_perturb = random.randint(3, 10)
        
        # We need a helper to make a single valid random move
        def make_random_move(rows, used_diffs, row_diffs, row_maxes):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            old_val = rows[r_idx][pos]
            lower_bound = rows[r_idx][pos - 1] + 1
            upper_bound = rows[r_idx][pos + 1] - 1 if pos < k else max(rows[r_idx]) + 50
            
            if upper_bound < lower_bound:
                return False
            
            # Try to find a valid new value
            # Randomly sample a few candidates
            for _ in range(20):
                new_val = random.randint(lower_bound, upper_bound)
                if new_val == old_val:
                    continue
                
                new_row = rows[r_idx][:]
                new_row[pos] = new_val
                
                new_diffs = set()
                n = len(new_row)
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        d = new_row[j] - new_row[i]
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                forbidden = used_diffs - row_diffs[r_idx]
                if new_diffs & forbidden:
                    continue
                
                # Valid move
                used_diffs.difference_update(row_diffs[r_idx])
                used_diffs.update(new_diffs)
                rows[r_idx] = new_row
                row_diffs[r_idx] = new_diffs
                row_maxes[r_idx] = max(new_row)
                return True
            return False

        for _ in range(n_perturb):
            if time.time() > deadline - 10:
                break
            make_random_move(current_rows, current_used_diffs, current_row_diffs, current_row_maxes)
        
        # Reset temperature for the next SA cycle
        T = 100.0

    # Final Polish: A quick greedy push-down to clean up
    # This is a deterministic local search that only accepts improvements
    # It's fast and can squeeze out the last bit of scope
    def greedy_polish(rows, used_diffs, row_diffs, row_maxes, time_limit=5.0):
        start = time.time()
        progress = True
        while progress and time.time() - start < time_limit:
            progress = False
            # Randomize row order
            order = list(range(n_rows))
            random.shuffle(order)
            
            for r_idx in order:
                if time.time() - start > time_limit:
                    break
                row = rows[r_idx]
                # Try to lower each position from right to left
                for pos in range(k, 0, -1):
                    old_val = row[pos]
                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1 # Only try to lower
                    if pos < k:
                        upper_bound = min(upper_bound, row[pos + 1] - 1)
                    
                    if upper_bound < lower_bound:
                        continue
                    
                    # Try smallest valid value
                    found = False
                    for new_val in range(lower_bound, upper_bound + 1):
                        new_row = row[:]
                        new_row[pos] = new_val
                        
                        new_diffs = set()
                        n = len(new_row)
                        valid = True
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = new_row[j] - new_row[i]
                                if d in new_diffs:
                                    valid = False
                                    break
                                new_diffs.add(d)
                            if not valid:
                                break
                        
                        if not valid:
                            continue
                        
                        forbidden = used_diffs - row_diffs[r_idx]
                        if new_diffs & forbidden:
                            continue
                        
                        # Valid and lower
                        used_diffs.difference_update(row_diffs[r_idx])
                        used_diffs.update(new_diffs)
                        rows[r_idx] = new_row
                        row_diffs[r_idx] = new_diffs
                        row_maxes[r_idx] = max(new_row)
                        progress = True
                        found = True
                        break # Move to next position
                    if found:
                        continue # Continue to next position in this row
                    else:
                        break # No more improvements in this row? 
                              # Actually, if pos k didn't work, pos k-1 might still work?
                              # But lowering pos k-1 makes it harder for pos k.
                              # So if we can't lower pos k, we might still lower pos k-1 
                              # if it doesn't affect pos k's validity? 
                              # Wait, lowering pos k-1 reduces the gap for pos k, 
                              # making it *harder* to find a valid pos k.
                              # So if we can't lower pos k, we probably can't lower pos k-1 
                              # in a way that helps the global max if pos k is the max.
                              # However, if pos k is NOT the max, lowering pos k-1 might help.
                              # Let's just break to be safe and fast.
                        pass

        return rows, used_diffs, row_diffs, row_maxes, max(row_maxes)

    # Run final polish on the best solution found
    if time.time() < deadline - 2:
        polish_rows = [r[:] for r in best_solution]
        polish_used = set(best_used_diffs)
        polish_diffs = [get_row_diffs(r) for r in polish_rows]
        polish_maxes = [max(r) for r in polish_rows]
        
        polish_rows, polish_used, polish_diffs, polish_maxes, polish_score = \
            greedy_polish(polish_rows, polish_used, polish_diffs, polish_maxes, 5.0)
        
        if polish_score < best_score:
            best_solution = polish_rows
            best_score = polish_score

    return best_solution

# Main entry point check
if __name__ == "__main__":
    # Example usage for testing
    # n_rows = 7
    # k = 5
    # res = construct(n_rows, k)
    # print(res)
    pass
