import time
import random
import math

def construct(n_rows, k):
    """
    Finds a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimizes the largest entry.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # Helper to check if a new value v at row r, index j is valid
    # given the current rows and used_diffs
    # We will maintain used_diffs explicitly for speed
    
    def greedy_construct():
        """
        Constructs a valid solution using a greedy approach.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try next integer
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        return rows, used_diffs

    # Initial solution
    best_rows, best_used_diffs = greedy_construct()
    best_max = max(max(row) for row in best_rows)
    
    # If k is small or n is small, greedy might be optimal.
    # But we try to improve.
    
    def get_max(rows):
        return max(max(row) for row in rows)

    def try_improve(rows, used_diffs, current_max, iterations):
        """
        Runs Simulated Annealing / Hill Climbing to improve the solution.
        Returns improved rows, used_diffs, and max.
        """
        rows = [r[:] for r in rows]
        used_diffs = set(used_diffs)
        current_max = current_max
        
        # Temperature for SA
        T = 5.0
        cool_rate = 0.99
        
        # Precompute indices for speed? No, dynamic is fine.
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            # Pick a random row and index
            r_idx = random.randint(0, n_rows - 1)
            # Pick an index > 0 (since 0 is fixed)
            j_idx = random.randint(1, k)
            
            old_v = rows[r_idx][j_idx]
            
            # Determine valid range for new value
            low = rows[r_idx][j_idx - 1] + 1
            high = rows[r_idx][j_idx + 1] - 1 if j_idx < k else current_max + 10 # Allow expansion? No, we want to minimize.
            # Actually, if we expand, we might get stuck. Let's restrict to [low, old_v] first.
            # If we can't reduce, we might need to increase to allow other reductions?
            # Let's stick to reducing or swapping within current bounds first.
            
            if low > old_v:
                continue
                
            # Try a random value in [low, old_v]
            # Bias towards smaller values?
            new_v = random.randint(low, old_v)
            
            if new_v == old_v:
                continue

            # Check validity
            # Remove old diffs
            old_diffs = set()
            for i in range(j_idx):
                old_diffs.add(old_v - rows[r_idx][i])
            for i in range(j_idx + 1, len(rows[r_idx])):
                old_diffs.add(rows[r_idx][i] - old_v)
            
            # Temporarily remove
            for d in old_diffs:
                used_diffs.discard(d)
                
            # Check new diffs
            new_diffs = set()
            valid = True
            for i in range(j_idx):
                d = new_v - rows[r_idx][i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                for i in range(j_idx + 1, len(rows[r_idx])):
                    d = rows[r_idx][i] - new_v
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
            
            if not valid:
                # Restore
                used_diffs.update(old_diffs)
                continue

            # Valid move
            # Calculate new max
            # If new_v is the max and we reduced it, max might decrease.
            # If new_v is not the max, max stays same or increases (if we increased, but we didn't).
            # We only pick new_v <= old_v.
            
            # Update state
            rows[r_idx][j_idx] = new_v
            used_diffs.update(new_diffs)
            
            # Determine if this is an improvement
            # We don't track global max efficiently, so we just accept if valid.
            # To ensure we find the global min, we should track the max.
            # For now, accept all valid moves (Hill Climbing with random restarts/steps).
            # This might wander.
            
            # Cooling
            T *= cool_rate

        return rows, used_diffs, get_max(rows)

    # Main loop: Multiple restarts/improvement phases
    # Since greedy is deterministic, we need randomization to explore.
    # We can randomize the greedy construction by shuffling the "order" of difference checks?
    # Or simply, we can perturb the greedy result.
    
    # Let's do:
    # 1. Greedy
    # 2. Improve
    # 3. If stuck, Perturb (Randomly change a few elements to valid values, even if larger)
    # 4. Improve
    
    while time.time() < deadline:
        # Improve current best
        rows, used_diffs, curr_max = try_improve(best_rows, best_used_diffs, best_max, 1000)
        
        if curr_max < best_max:
            best_max = curr_max
            best_rows = rows
            best_used_diffs = used_diffs
        
        # If we haven't improved in a while, restart from a perturbed state or new greedy?
        # Since greedy is fast, let's just run more improvements.
        # To avoid local minima, we can occasionally "kick" the solution.
        # Kick: Pick a random element and change it to a random valid value in a wider range.
        
        # Let's add a "kick" phase if no improvement in last 10000 iterations (approx)
        # For simplicity, let's just keep improving. The random moves in try_improve act as the search.
        
        # If we want to be safer, we can try a different initialization strategy.
        # Randomized Greedy:
        # In greedy, instead of x = last + 1, we can try x = last + 1 + random(0, 5) if valid?
        # No, that makes it worse.
        
        # Let's rely on the SA-like moves in try_improve.
        # To ensure we don't get stuck in a very bad local minimum,
        # we can occasionally revert to a "loose" greedy and then tighten.
        
        if random.random() < 0.1:
            # Occasionally restart from a fresh greedy to escape deep local minima
            # (Though greedy is deterministic, so it's the same. 
            # We need a randomized greedy.)
            
            def randomized_greedy():
                used_diffs = set()
                rows = []
                for _ in range(n_rows):
                    row = [0]
                    while len(row) < k + 1:
                        # Try candidates starting from last+1
                        # With some probability, skip the first valid one?
                        # No, let's just use the standard greedy for now.
                        # The improvement phase does the heavy lifting.
                        x = row[-1] + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used_diffs:
                                row.append(x)
                                used_diffs.update(new_diffs)
                                break
                            x += 1
                    rows.append(row)
                return rows, used_diffs

            # Since randomized_greedy is same as greedy, this restart is useless.
            # Let's implement a true random restart by perturbing the current best.
            # Perturbation: Pick 3-5 random elements and change them to random valid values 
            # in a range [low, high] where high is current_max + 10.
            
            rows_perturbed = [r[:] for r in best_rows]
            used_diffs_perturbed = set(best_used_diffs)
            
            # Perform a few random valid moves (allowing increase in max)
            for _ in range(10):
                r_idx = random.randint(0, n_rows - 1)
                j_idx = random.randint(1, k)
                old_v = rows_perturbed[r_idx][j_idx]
                
                # Remove old diffs
                old_diffs = set()
                for i in range(j_idx):
                    old_diffs.add(old_v - rows_perturbed[r_idx][i])
                for i in range(j_idx + 1, len(rows_perturbed[r_idx])):
                    old_diffs.add(rows_perturbed[r_idx][i] - old_v)
                
                for d in old_diffs:
                    used_diffs_perturbed.discard(d)
                
                # Pick a new random value
                # Range: [prev+1, next-1]
                low = rows_perturbed[r_idx][j_idx - 1] + 1
                high = rows_perturbed[r_idx][j_idx + 1] - 1 if j_idx < k else best_max + 20
                
                if low > high:
                    # Restore
                    used_diffs_perturbed.update(old_diffs)
                    continue
                    
                # Try a few random values
                found = False
                for _ in range(20):
                    new_v = random.randint(low, high)
                    if new_v == old_v:
                        continue
                    new_diffs = set()
                    valid = True
                    for i in range(j_idx):
                        d = new_v - rows_perturbed[r_idx][i]
                        if d in used_diffs_perturbed:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        for i in range(j_idx + 1, len(rows_perturbed[r_idx])):
                            d = rows_perturbed[r_idx][i] - new_v
                            if d in used_diffs_perturbed:
                                valid = False
                                break
                            new_diffs.add(d)
                    
                    if valid:
                        rows_perturbed[r_idx][j_idx] = new_v
                        used_diffs_perturbed.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    # Restore
                    used_diffs_perturbed.update(old_diffs)

            # Now improve the perturbed solution
            rows_imp, used_diffs_imp, curr_max_imp = try_improve(rows_perturbed, used_diffs_perturbed, get_max(rows_perturbed), 1000)
            
            if curr_max_imp < best_max:
                best_max = curr_max_imp
                best_rows = rows_imp
                best_used_diffs = used_diffs_imp

    return best_rows

