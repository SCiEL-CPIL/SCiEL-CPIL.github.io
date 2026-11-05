"""
Strategy: Simulated Annealing over gap representation.
State: List of lists of gaps. Row r has k gaps (g_1..g_k), where a_i = sum(g_j for j<=i).
Moves: 
1. Swap two gaps in the same row.
2. Add a random integer to a gap.
3. Subtract a random integer from a gap (if result > 0).
Cost: 
- Primary: Max entry (scope)
- Secondary: Number of duplicate differences (conflicts)
Objective: Minimize scope, then minimize conflicts.
"""
import random
import time
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max entry
    lower_bound = total_diffs
    
    # Helper to compute row entries from gaps
    def gaps_to_row(gaps):
        row = [0]
        curr = 0
        for g in gaps:
            curr += g
            row.append(curr)
        return row
    
    # Helper to compute all differences for a row
    def row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    # Initial state: Greedy construction to get a valid starting point
    def initial_greedy():
        used_diffs = set()
        rows_gaps = []
        for _ in range(n_rows):
            row = [0]
            gaps = []
            x = 0
            while len(row) < k + 1:
                # Try small increments first, then larger
                found = False
                for delta in range(1, 100):
                    x_candidate = x + delta
                    # Check conflicts with current row
                    row_candidate = row + [x_candidate]
                    new_diffs = set()
                    for i in range(len(row_candidate)):
                        for j in range(i + 1, len(row_candidate)):
                            d = row_candidate[j] - row_candidate[i]
                            if d in used_diffs:
                                break
                            new_diffs.add(d)
                        else:
                            continue
                        break
                    else:
                        # No conflicts
                        row = row_candidate
                        gaps.append(delta)
                        x = x_candidate
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    # If we get stuck, just pick a large random jump
                    # This might create conflicts, but SA will fix them
                    delta = random.randint(1, 50)
                    x += delta
                    row.append(x)
                    gaps.append(delta)
                    # Update used_diffs (might have conflicts, but we proceed)
                    # For initialization, we want a "reasonable" start, 
                    # so let's restart this row if it gets too messy
                    # Actually, simple greedy without conflict check for init is bad.
                    # Let's use a simple random valid start.
                    pass
            
            # If the above loop failed to produce k gaps, use a simple random valid set
            if len(gaps) < k:
                # Fallback: random distinct differences
                # Just pick random gaps that are large enough
                g = [random.randint(1, 10) for _ in range(k)]
                rows_gaps.append(g)
            else:
                rows_gaps.append(g)
        return rows_gaps

    # Better initialization: Random valid rows using backtracking with a small limit
    def random_valid_row(used_diffs, rng):
        row = [0]
        gaps = []
        x = 0
        for _ in range(k):
            found = False
            # Try a range of values
            # To ensure validity, we might need to retry
            for _ in range(100):
                # Pick a random increment
                # Biased towards smaller to keep scope low
                delta = rng.randint(1, 20)
                x_candidate = x + delta
                # Check conflicts
                valid = True
                new_diffs = set()
                for prev in row:
                    d = x_candidate - prev
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x_candidate)
                    gaps.append(delta)
                    x = x_candidate
                    used_diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                # If we can't find a valid increment, we return None to signal failure
                # In that case, the caller should retry with a new random seed
                return None
        return gaps

    def generate_initial_solution():
        rng = random.Random()
        for _ in range(50):
            used_diffs = set()
            rows_gaps = []
            success = True
            for _ in range(n_rows):
                g = random_valid_row(used_diffs, rng)
                if g is None:
                    success = False
                    break
                rows_gaps.append(g)
            if success:
                return rows_gaps
        # Fallback: simple random
        return [[random.randint(1, 10) for _ in range(k)] for _ in range(n_rows)]

    current_gaps = generate_initial_solution()
    
    # Evaluate current state
    def evaluate(gaps):
        rows = [gaps_to_row(g) for g in gaps]
        max_entry = max(max(r) for r in rows)
        
        all_diffs = []
        for r in rows:
            all_diffs.extend(row_diffs(r))
        
        # Count duplicates
        if not all_diffs:
            return max_entry, 0
        
        # Efficient duplicate count
        # We can use a dict to count occurrences
        counts = {}
        for d in all_diffs:
            counts[d] = counts.get(d, 0) + 1
        
        conflicts = sum(c - 1 for c in counts.values() if c > 1)
        
        return max_entry, conflicts

    current_max, current_conflicts = evaluate(current_gaps)
    best_gaps = [g[:] for g in current_gaps]
    best_max, best_conflicts = current_max, current_conflicts
    
    # Simulated Annealing
    T = 500.0
    cooling = 0.9999
    min_T = 0.1
    
    # We want to minimize (max_entry, conflicts)
    # We can combine them into a single cost function: cost = max_entry * 1000 + conflicts
    # Or use lexicographic ordering in the acceptance criterion.
    
    def get_cost(gaps):
        m, c = evaluate(gaps)
        return m * 1000 + c

    current_cost = get_cost(current_gaps)
    best_cost = current_cost
    
    step = 0
    while T > min_T and time.time() < deadline:
        # Generate a neighbor
        new_gaps = [g[:] for g in current_gaps]
        
        # Random move
        move_type = random.randint(0, 2)
        r_idx = random.randint(0, n_rows - 1)
        
        if move_type == 0:
            # Swap two gaps in row r_idx
            if k >= 2:
                i, j = random.sample(range(k), 2)
                new_gaps[r_idx][i], new_gaps[r_idx][j] = new_gaps[r_idx][j], new_gaps[r_idx][i]
        elif move_type == 1:
            # Add to a gap
            i = random.randint(0, k - 1)
            delta = random.randint(1, 5)
            new_gaps[r_idx][i] += delta
        else:
            # Subtract from a gap
            i = random.randint(0, k - 1)
            delta = random.randint(1, 5)
            if new_gaps[r_idx][i] > delta:
                new_gaps[r_idx][i] -= delta
            else:
                # Invalid move, skip
                continue
        
        new_cost = get_cost(new_gaps)
        
        # Acceptance criterion
        if new_cost < current_cost:
            current_gaps = new_gaps
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_gaps = [g[:] for g in current_gaps]
                # Update best_max and best_conflicts for tracking
                best_max, best_conflicts = evaluate(best_gaps)
        else:
            # Delta cost
            delta_cost = new_cost - current_cost
            # Accept with probability exp(-delta_cost / T)
            # Since delta_cost > 0, we use -delta_cost/T
            if delta_cost < 1000: # Avoid overflow in exp
                p = math.exp(-delta_cost / T)
                if random.random() < p:
                    current_gaps = new_gaps
                    current_cost = new_cost
        
        T *= cooling
        step += 1
        
        # Optional: If we hit the lower bound, we can stop early
        if best_max == lower_bound:
            break
    
    # Convert best_gaps back to rows
    best_rows = [gaps_to_row(g) for g in best_gaps]
    
    return best_rows
# EVOLVE-BLOCK-END
