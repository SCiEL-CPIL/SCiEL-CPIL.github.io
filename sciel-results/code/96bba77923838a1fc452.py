"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry while keeping all pairwise differences distinct.
    """
    deadline = time.time() + 110  # 10s safety margin
    
    # Parameters
    target_size = k + 1
    max_iter = 500000
    
    # Helper to compute cost (number of duplicate differences)
    def get_cost(rows):
        diffs = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in diffs:
                        diffs[d] += 1
                    else:
                        diffs[d] = 1
        # Cost is the sum of (count - 1) for all duplicates, or simply sum(count) - unique_count
        # Actually, we want 0 if all unique. 
        # Let's just count how many differences are shared.
        # If a diff appears c times, it contributes c-1 to the "conflict" count.
        # Total conflicts = sum(count) - len(diffs)
        return sum(counts - 1 for counts in diffs.values() if counts > 1)
    
    def get_max_entry(rows):
        return max(max(r) for r in rows)
    
    # Initialize: Random valid rows (increasing, start 0)
    # We start with a somewhat tight range to encourage small max entries
    # But we need to ensure they are valid increasing sequences.
    # A good initial guess is random integers in [0, 150] sorted.
    # However, we must ensure strictly increasing.
    
    def random_row(max_val):
        # Pick k values from 1..max_val, sort, prepend 0
        vals = random.sample(range(1, max_val + 1), k)
        vals.sort()
        return [0] + vals

    # Initial population: a single solution
    # Start with a range slightly larger than the counting bound to allow movement
    # Counting bound is ~105. Best known 112.
    # Let's start with max_val = 130
    
    current_max_val = 140 
    current_rows = [random_row(current_max_val) for _ in range(n_rows)]
    
    # We need to be careful: random_row might produce duplicates if k > max_val, 
    # but here k=6, max_val=140, so it's fine.
    # Also, the problem says "increasing integers". Random sample ensures distinctness.
    
    current_cost = get_cost(current_rows)
    current_max = get_max_entry(current_rows)
    
    best_rows = [list(r) for r in current_rows]
    best_max = current_max
    
    temperature = 50.0
    cooling_rate = 0.999
    
    # Local search moves:
    # 1. Pick a random row, pick a random index (1..k), change its value.
    #    To maintain increasing order, we can pick a new value and re-sort?
    #    No, the row must start at 0 and increase.
    #    Better: Pick a row i. Pick a position j in that row.
    #    Generate a new value for r[i][j] that keeps the row strictly increasing.
    #    Lower bound: r[i][j-1] + 1
    #    Upper bound: r[i][j+1] - 1 (or current_max_val if j is last)
    
    while time.time() < deadline:
        # Pick a move
        row_idx = random.randint(0, n_rows - 1)
        # Pick a position to change. Don't change index 0 (always 0).
        pos_idx = random.randint(1, k)
        
        row = current_rows[row_idx]
        
        # Determine bounds for new value
        lower = row[pos_idx - 1] + 1
        upper = row[pos_idx + 1] - 1 if pos_idx < k else current_max_val
        
        if lower > upper:
            # Cannot change without violating order?
            # This can happen if the row is very tight.
            # In that case, maybe try to expand the max_val or change a different position?
            # For simplicity, skip this move or try to shift the whole row?
            # Let's just skip.
            continue
            
        # Randomly choose a new value
        # To allow exploration, sometimes pick a value outside the current tight range?
        # No, the bounds are defined by neighbors.
        # However, if we want to increase the max entry, we might need to allow upper > current_max_val.
        # But we are minimizing max entry.
        # Let's allow upper to be max(upper, current_max_val) to allow growth if needed?
        # Actually, if we want to reduce max entry, we shouldn't grow it.
        # But if stuck, we might need to grow to escape local minima.
        # Let's allow upper to be current_max_val + 10 to allow some expansion.
        upper = max(upper, current_max_val + 5)
        
        new_val = random.randint(lower, upper)
        
        if new_val == row[pos_idx]:
            continue
            
        # Calculate new cost
        old_val = row[pos_idx]
        row[pos_idx] = new_val
        
        # Check if row is still strictly increasing (it should be by bounds)
        # But wait, if pos_idx < k, and new_val > row[pos_idx+1], we violated upper bound?
        # We set upper = row[pos_idx+1] - 1. So it's safe.
        
        new_cost = get_cost(current_rows)
        new_max = get_max_entry(current_rows)
        
        # Acceptance criterion
        # We want to minimize cost first, then max entry.
        # Primary objective: Cost (duplicates). Must be 0 for valid solution.
        # Secondary: Max entry.
        
        if new_cost < current_cost:
            current_cost = new_cost
            current_max = new_max
            if current_cost == 0:
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in current_rows]
        elif new_cost == current_cost:
            # Same cost, prefer lower max entry
            if new_max < current_max:
                current_max = new_max
                if current_cost == 0 and current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in current_rows]
            else:
                # Accept with some probability to escape plateaus?
                # If cost is 0, we are in the valid region. We want to minimize max.
                # If cost > 0, we are invalid. We should accept moves that don't increase cost too much?
                # Simulated annealing on cost + lambda * max?
                # Let's use a combined score: cost * 1000 + max_entry
                old_score = current_cost * 1000 + current_max
                new_score = new_cost * 1000 + new_max
                delta = new_score - old_score
                if delta < 0:
                    current_cost = new_cost
                    current_max = new_max
                    if current_cost == 0 and current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in current_rows]
                else:
                    # Accept with probability exp(-delta / temperature)
                    if random.random() < pow(sys.e, -delta / temperature):
                        current_cost = new_cost
                        current_max = new_max
                        if current_cost == 0 and current_max < best_max:
                            best_max = current_max
                            best_rows = [list(r) for r in current_rows]
                    else:
                        # Reject: revert
                        row[pos_idx] = old_val
        else:
            # Cost increased
            # Accept with probability exp(-delta / temperature)
            old_score = current_cost * 1000 + current_max
            new_score = new_cost * 1000 + new_max
            delta = new_score - old_score
            if random.random() < pow(sys.e, -delta / temperature):
                current_cost = new_cost
                current_max = new_max
                # Don't update best if cost > 0
            else:
                row[pos_idx] = old_val
                
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1
            
    return best_rows

# EVOLVE-BLOCK-END
