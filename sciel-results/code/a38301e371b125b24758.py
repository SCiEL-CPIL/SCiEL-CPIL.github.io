import time
import random
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses Simulated Annealing to escape local minima found by greedy methods.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Greedy baseline
    def greedy_construct(n_rows, k, rng=None):
        if rng is None:
            rng = random.Random(42)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            # Add some randomness to the greedy choice if possible?
            # Standard greedy picks the smallest valid x.
            # We can try picking from a small window of valid x's.
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def get_diffs(rows):
        """Returns a dictionary of difference -> count"""
        diff_count = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
        return diff_count

    def calculate_cost(rows):
        """
        Returns (max_val, collisions).
        We want to minimize max_val, but collisions must be 0 for validity.
        """
        max_val = 0
        for row in rows:
            if row:
                if row[-1] > max_val:
                    max_val = row[-1]
        
        diff_count = get_diffs(rows)
        collisions = 0
        for cnt in diff_count.values():
            if cnt > 1:
                collisions += cnt - 1
                
        # Cost priority: 1. Minimize collisions. 2. Minimize max_val.
        # We return a tuple that can be compared lexicographically.
        # Lower is better.
        return (collisions, max_val)

    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            # Check strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Initial solution
    rng = random.Random(42)
    best_rows = greedy_construct(n_rows, k)
    best_cost = calculate_cost(best_rows)
    
    # Simulated Annealing
    current_rows = [list(r) for r in best_rows]
    current_cost = best_cost
    
    # Temperature parameters
    T = 100.0
    T_min = 0.01
    alpha = 0.999
    
    steps = 0
    max_steps = 200000 # Will be limited by time anyway
    
    # Pre-compute indices for faster access?
    # n_rows is small (4), k is small (7).
    
    while time.time() < deadline - 2.0 and T > T_min:
        steps += 1
        if steps % 1000 == 0:
            # Check time strictly
            if time.time() > deadline - 2.0:
                break

        # Choose a move
        # 1. Pick a row
        ri = rng.randrange(n_rows)
        # 2. Pick a position in that row (1 to k, index 0 is fixed at 0)
        pos = rng.randrange(1, k + 1)
        
        # Determine move type
        move_type = rng.random()
        
        old_val = current_rows[ri][pos]
        new_val = old_val
        
        if move_type < 0.7:
            # Shift: small change
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
        elif move_type < 0.9:
            # Large shift: jump to a random nearby value
            # Try to move towards the average or just random within a range
            # Let's try to shrink if we are large, or expand if stuck?
            # Just random in [1, old_val * 1.5]
            new_val = rng.randint(1, max(1, old_val + 10))
        else:
            # Swap with neighbor? No, must maintain order.
            # Instead, let's try to replace the value with the smallest valid value
            # that is greater than the previous element.
            # This is a "greedy repair" move.
            prev_val = current_rows[ri][pos - 1]
            # Find smallest x > prev_val such that diffs are valid
            # This is expensive to check fully, but we can check against current diff counts
            # Actually, let's just do a larger random jump.
            new_val = old_val + rng.randint(-20, 20)
            
        # Validate move locally (order constraint)
        if new_val <= current_rows[ri][pos - 1]:
            new_val = current_rows[ri][pos - 1] + 1
        if pos < k and new_val >= current_rows[ri][pos + 1]:
            new_val = current_rows[ri][pos + 1] - 1
            
        if new_val <= current_rows[ri][pos - 1]:
            continue # Cannot satisfy order
        
        # Apply move
        current_rows[ri][pos] = new_val
        
        # Calculate new cost
        new_cost = calculate_cost(current_rows)
        
        # Acceptance probability
        delta_cost = new_cost[0] * 10000 + new_cost[1] - (current_cost[0] * 10000 + current_cost[1])
        
        if delta_cost < 0:
            # Better
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [list(r) for r in current_rows]
        else:
            # Worse
            # Probability = exp(-delta_cost / T)
            # Since delta_cost can be large, we cap it to avoid underflow
            prob = 1.0
            if T > 0:
                # Use a scaled delta to keep probabilities reasonable
                # Collisions are weighted heavily (10000)
                # So a collision change is a huge penalty.
                # We want to allow moving into collisions if T is high.
                scaled_delta = delta_cost / 100.0 
                if scaled_delta < 100: # Avoid exp(-large)
                    prob = 2.718281828 ** (-scaled_delta / T)
                else:
                    prob = 0.0
            if rng.random() < prob:
                current_cost = new_cost
            else:
                # Reject
                current_rows[ri][pos] = old_val
                
        # Cool down
        T *= alpha
        
        # Occasional restart of a row to escape deep minima
        if steps % 5000 == 0:
             # Pick the row with the largest max
             target_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
             # Re-greedy this row based on other rows
             other_rows = [r for i, r in enumerate(current_rows) if i != target_row_idx]
             used_diffs = set()
             for r in other_rows:
                 for i in range(len(r)):
                     for j in range(i + 1, len(r)):
                         used_diffs.add(r[j] - r[i])
             
             new_row = [0]
             x = 0
             while len(new_row) < k + 1:
                 x += 1
                 new_diffs = {x - a for a in new_row}
                 if not (new_diffs & used_diffs):
                     new_row.append(x)
             
             # Compare costs
             old_row = current_rows[target_row_idx]
             current_rows[target_row_idx] = new_row
             new_cost = calculate_cost(current_rows)
             
             # Accept if better or with some probability?
             # Let's just accept if it doesn't increase collisions significantly
             if new_cost[0] <= current_cost[0]:
                 current_cost = new_cost
                 if new_cost < best_cost:
                     best_cost = new_cost
                     best_rows = [list(r) for r in current_rows]
             else:
                 # Revert
                 current_rows[target_row_idx] = old_row

    # Final Polish: Try to reduce max entry further with local greedy adjustments
    # Iterate through rows and try to shift elements down if possible
    final_rows = [list(r) for r in best_rows]
    
    for _ in range(1000): # Limited polish steps
        if time.time() > deadline - 1.0:
            break
        improved = False
        for ri in range(n_rows):
            row = final_rows[ri]
            # Try to decrease each element from right to left
            for pos in range(k, 0, -1):
                old_val = row[pos]
                # Try to decrease
                # Find smallest valid value > row[pos-1]
                # We need to check diffs against all other rows and the rest of this row
                # This is O(N*K^2) per check, which is fine for small N, K
                # Let's try a few steps down
                for new_val in range(old_val - 1, row[pos-1], -1):
                    # Check if new_val is valid
                    # 1. Order: new_val > row[pos-1] (guaranteed by range)
                    # 2. Diffs
                    # Calculate diffs involving this new element
                    # Diffs with previous elements in this row
                    local_diffs = {new_val - row[i] for i in range(pos)}
                    # Diffs with next elements in this row
                    local_diffs |= {row[j] - new_val for j in range(pos+1, k+1)}
                    
                    # Check against global used diffs
                    # We need a global set of diffs for the other rows and the rest of this row
                    # This is expensive to recompute every time.
                    # Simplified check: 
                    # If we just check if any of these diffs are in the "forbidden" set.
                    # Let's build the forbidden set once per row iteration?
                    # Actually, let's just use the full validate/cost function for simplicity 
                    # if we make a change, but that's slow.
                    
                    # Heuristic: Just try the immediate next valid value?
                    # For now, let's skip this complex polish to save time and rely on SA.
                    break
                if new_val != old_val:
                    # Validate full solution?
                    # For speed, let's just update and check cost at the end of the row loop
                    row[pos] = new_val
                    improved = True
                    break # Restart row check from right? Or just continue?
            # After modifying a row, check if solution is still valid
            if not validate(final_rows, n_rows, k):
                # Revert if invalid? Or just stop polishing?
                # Reverting is safer.
                final_rows = [list(r) for r in best_rows]
                break
            else:
                # Check if max improved
                new_max = max(max(r) for r in final_rows)
                old_max = max(max(r) for r in best_rows)
                if new_max < old_max:
                    best_rows = [list(r) for r in final_rows]
                # If not improved, keep going but don't update best yet

    # Ensure the returned result is valid
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
        
    return best_rows

# EVOLVE-BLOCK-END
