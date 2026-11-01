import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    row_len = k + 1
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def get_all_diffs(rows):
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.append(r[j] - r[i])
        return diffs

    def check_validity_and_scope(rows):
        diffs = get_all_diffs(rows)
        if len(diffs) != len(set(diffs)):
            return False, get_scope(rows)
        return True, get_scope(rows)

    def greedy_init():
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                # Try small increments first
                found = False
                x = last + 1
                while x < 500:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        found = True
                        break
                    x += 1
                
                if not found:
                    # Fallback: just pick a large random number
                    x = last + random.randint(10, 100)
                    # Ensure it doesn't collide (unlikely with large random)
                    while True:
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            break
                        x += 1
                    row.append(x)
                    for a in row[:-1]:
                        used_diffs.add(x - a)
                    
            rows.append(row)
            
        return rows

    def cost_function(rows, target_scope):
        """
        Returns a cost value. Lower is better.
        If valid and max <= target_scope, cost is max_element.
        If invalid or max > target_scope, cost is high.
        """
        diffs = get_all_diffs(rows)
        diff_counts = {}
        for d in diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        collisions = 0
        for count in diff_counts.values():
            if count > 1:
                collisions += count - 1
        
        max_elem = max(max(r) for r in rows)
        
        # Heuristic:
        # 1. Strong penalty for collisions
        # 2. Strong penalty for exceeding target_scope
        # 3. Weak penalty for max_elem to encourage smaller values
        
        collision_penalty = collisions * 1000.0
        scope_excess = max(0, max_elem - target_scope)
        scope_penalty = scope_excess * 100.0
        
        return collision_penalty + scope_penalty + max_elem

    def simulated_annealing(rows, target_scope, time_limit, seed):
        random.seed(seed)
        start = time.time()
        
        current_rows = [r[:] for r in rows]
        current_cost = cost_function(current_rows, target_scope)
        
        best_rows = [r[:] for r in current_rows]
        best_cost = current_cost
        
        # Initial temperature
        T = 100.0
        cooling_rate = 0.999
        
        while time.time() - start < time_limit:
            # Pick a random row and position to change
            r_idx = random.randint(0, n_rows - 1)
            pos_idx = random.randint(1, row_len - 1) # Don't change the 0
            
            # Pick a new value
            # Strategy: change by a small delta
            old_val = current_rows[r_idx][pos_idx]
            
            # Ensure increasing order
            min_val = current_rows[r_idx][pos_idx - 1] + 1
            max_val = current_rows[r_idx][pos_idx + 1] - 1 if pos_idx < row_len - 1 else target_scope + 100
            
            if min_val > max_val:
                # If stuck, try a larger jump or reset this element
                # For now, skip or pick a random valid range
                if random.random() < 0.5:
                    continue
                else:
                    # Try to fix by picking a value far away
                    new_val = random.randint(min_val, max(min_val + 100, target_scope + 50))
            else:
                # Bias towards smaller values
                if random.random() < 0.8:
                    new_val = random.randint(min_val, min(max_val, min_val + 20))
                else:
                    new_val = random.randint(min_val, max_val)
            
            # Apply change
            current_rows[r_idx][pos_idx] = new_val
            
            # Check if strictly increasing is violated
            if pos_idx > 0 and current_rows[r_idx][pos_idx] <= current_rows[r_idx][pos_idx - 1]:
                current_rows[r_idx][pos_idx] = old_val
                continue
            if pos_idx < row_len - 1 and current_rows[r_idx][pos_idx] >= current_rows[r_idx][pos_idx + 1]:
                current_rows[r_idx][pos_idx] = old_val
                continue

            new_cost = cost_function(current_rows, target_scope)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows = [r[:] for r in current_rows]
                    # If we found a valid solution within target, we can lower T faster?
                    # Or just continue
            else:
                # Revert
                current_rows[r_idx][pos_idx] = old_val
            
            T *= cooling_rate
            if T < 1e-5:
                T = 1e-5

        return best_rows, best_cost

    def perturb(rows, intensity=3):
        """
        Randomly changes a few elements to escape local minima.
        """
        perturbed = [r[:] for r in rows]
        for _ in range(intensity):
            r_idx = random.randint(0, n_rows - 1)
            pos_idx = random.randint(1, row_len - 1)
            
            old_val = perturbed[r_idx][pos_idx]
            # Change by a random amount
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            
            # Ensure increasing order
            min_val = perturbed[r_idx][pos_idx - 1] + 1
            max_val = perturbed[r_idx][pos_idx + 1] - 1 if pos_idx < row_len - 1 else 10000
            
            if new_val < min_val:
                new_val = min_val + random.randint(0, 5)
            if new_val > max_val:
                new_val = max_val - random.randint(0, 5)
            
            if min_val <= new_val <= max_val:
                perturbed[r_idx][pos_idx] = new_val
        return perturbed

    # Phase 1: Greedy Init
    best_rows = greedy_init()
    best_scope = get_scope(best_rows)
    
    # Phase 2: ILS to minimize scope
    # We try to find a solution with scope < best_scope
    # We will iterate, lowering the target scope.
    
    current_target = best_scope
    
    # Run SA for a while to stabilize
    # Then try to lower target
    
    # We'll use a loop that attempts to find a valid solution with scope <= current_target - 1
    # If found, update best_scope and current_target.
    # If not, we might need to run SA on the current best to see if it can improve further 
    # (though if it's valid, cost is just max_elem, so SA will try to reduce max_elem).
    
    # Actually, let's just run ILS on the "minimize max_elem" objective directly,
    # but with a heavy penalty for invalidity.
    
    # Let's restart ILS a few times with different seeds and time budgets.
    
    seed_counter = 0
    
    # We will allocate time to several ILS runs.
    # Each run tries to minimize the scope.
    
    while time.time() < deadline - 5:
        # Run SA for a specific duration
        sa_time = 5.0
        if time.time() > deadline - 10:
            sa_time = deadline - time.time() - 2
        
        # Start from best_rows or a perturbed version
        start_rows = [r[:] for r in best_rows]
        
        # Perturb to explore
        if seed_counter > 0:
            start_rows = perturb(start_rows, intensity=2)
        
        # Run SA
        # Target scope is current best_scope - 1 to encourage improvement
        # But if we can't find valid, we might find a valid one with smaller scope anyway?
        # cost_function uses target_scope for penalties.
        # Let's set target_scope = best_scope - 1.
        # If it finds a valid solution, its max_elem will be <= best_scope - 1.
        
        target = best_scope - 1
        if target < 100: # Theoretical bound is 112, but let's not go too low if we are stuck
             # If we are stuck, maybe try to optimize current best?
             # For now, just try to find valid < best_scope
             pass

        sa_rows, sa_cost = simulated_annealing(start_rows, target, sa_time, seed=seed_counter)
        
        # Check if SA found a valid solution with smaller scope
        valid, scope = check_validity_and_scope(sa_rows)
        if valid and scope < best_scope:
            best_scope = scope
            best_rows = sa_rows
            # Reset target for next iteration
            # We don't need to update current_target variable explicitly, 
            # we use best_scope - 1 in the loop.
        
        seed_counter += 1
        
        # If we reached the theoretical bound, we can stop trying to go lower
        if best_scope <= 112:
             break

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    # Verify
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
