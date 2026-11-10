import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using Simulated Annealing.
    
    Args:
        n_rows: Number of rows.
        k: Number of differences per row (so row length is k+1).
        
    Returns:
        A list of n_rows rows, each a list of k+1 integers starting at 0.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def time_remaining():
        return deadline - time.time()

    def get_differences(rows):
        """Get a list of all differences and the max entry."""
        all_diffs = []
        max_entry = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.append(row[j] - row[i])
            if row:
                max_entry = max(max_entry, row[-1])
        return all_diffs, max_entry

    def count_conflicts(all_diffs):
        """Count the number of duplicate differences."""
        if not all_diffs:
            return 0
        # Using a set to count unique vs total
        unique_diffs = set(all_diffs)
        return len(all_diffs) - len(unique_diffs)

    def calculate_cost(rows):
        """
        Calculate the cost function.
        Cost = MaxEntry + Penalty * Conflicts.
        High penalty ensures we prefer valid solutions, but SA can explore invalid ones.
        """
        all_diffs, max_entry = get_differences(rows)
        conflicts = count_conflicts(all_diffs)
        # Penalty should be large enough that reducing conflicts is prioritized 
        # unless it saves a huge amount of span, but usually we want valid solutions.
        # However, for SA to work well, the cost landscape should be smooth.
        # Let's use a moderate penalty.
        penalty = 1000 
        return max_entry + penalty * conflicts

    def greedy_init(n_rows, k, rng):
        """
        Generate a random greedy initial solution.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Randomize the step selection slightly to avoid deterministic traps
            # We pick the smallest x such that differences are valid, 
            # but we can introduce randomness by picking from a small range of candidates
            # or just standard greedy with random restarts.
            
            # Standard greedy
            x = 1
            while len(row) < k + 1:
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
                    x += 1
                else:
                    x += 1
                if time.time() > deadline - 5:
                    break
            if len(row) == k + 1:
                rows.append(row)
            else:
                # Fallback if greedy got stuck (shouldn't happen for small k, but just in case)
                # Force complete
                last = row[-1]
                while len(row) < k + 1:
                    last += 1
                    row.append(last)
                rows.append(row)
        return rows

    def local_search_sa(rows, n_rows, k, rng, time_limit=10.0):
        """
        Simulated Annealing to improve the solution.
        """
        current_rows = [row[:] for row in rows]
        current_cost = calculate_cost(current_rows)
        best_rows = [row[:] for row in current_rows]
        best_cost = current_cost
        
        # SA Parameters
        T = 100.0  # Initial temperature
        cooling_rate = 0.995
        min_T = 1.0
        
        start_time = time.time()
        
        while T > min_T:
            if time.time() - start_time > time_limit:
                break
            if time.time() > deadline - 2:
                break
            
            # Generate a neighbor
            new_rows = [row[:] for row in current_rows]
            
            # Choose a move type
            move_type = rng.random()
            
            # Pick a random row and index
            r_idx = rng.randint(0, n_rows - 1)
            # Pick an index to modify (not the first one, which is 0)
            # We can modify any index from 1 to k
            c_idx = rng.randint(1, k)
            
            if move_type < 0.7:
                # Move 1: Shift value
                delta = rng.randint(-20, 20)
                if delta == 0:
                    continue
                new_val = new_rows[r_idx][c_idx] + delta
                # Ensure it stays positive (since row starts at 0 and increases)
                # Also ensure it doesn't violate ordering immediately? 
                # If we change a value, the row might not be sorted. 
                # Strategy: Change value, then sort the row. 
                # But we must ensure 0 remains the first element.
                # So, we remove the element, add new_val, and sort.
                # However, if new_val <= 0, it's invalid.
                if new_val <= 0:
                    continue
                
                # Extract row, modify, sort
                row = new_rows[r_idx]
                old_val = row[c_idx]
                row[c_idx] = new_val
                # Sort the row, but keep 0 at the front?
                # Actually, the problem says "increasing integers starting at 0".
                # So the set of numbers matters, and they must be sorted.
                # If we just sort the whole row, 0 will be first if it's the smallest.
                # Is it possible for new_val to be smaller than 0? No, we checked.
                # Is it possible for new_val to be smaller than other elements? Yes.
                # So we sort the row.
                row.sort()
                
            elif move_type < 0.9:
                # Move 2: Swap two elements in the same row
                if k >= 2:
                    c_idx2 = rng.randint(1, k)
                    if c_idx2 != c_idx:
                        row = new_rows[r_idx]
                        row[c_idx], row[c_idx2] = row[c_idx2], row[c_idx]
                        row.sort()
                    else:
                        continue
                else:
                    continue
            else:
                # Move 3: Randomize one element completely
                # Pick a random range for the new value
                # Max entry is usually around 100-200 for small k
                # Let's pick a value between 1 and current_max + 20
                current_max = max(new_rows[r_idx])
                upper_bound = current_max + 50
                new_val = rng.randint(1, upper_bound)
                
                row = new_rows[r_idx]
                row[c_idx] = new_val
                row.sort()

            # Calculate new cost
            new_cost = calculate_cost(new_rows)
            
            # Acceptance criterion
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
                current_rows = new_rows
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [row[:] for row in current_rows]
            
            T *= cooling_rate
            
        return best_rows, best_cost

    # 1. Initialize
    rng = random.Random(42)
    initial_rows = greedy_init(n_rows, k, rng)
    
    if not initial_rows:
        # Emergency fallback
        rows = []
        offset = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                row.append(offset + j * (k + 1))
            rows.append(row)
            offset += (k + 1) * (k + 1)
        return rows

    best_rows = initial_rows
    best_cost = calculate_cost(best_rows)
    
    # 2. Simulated Annealing with Restarts
    # We will run SA for a fixed time, then restart if stuck or time permits
    total_deadline = deadline - 5.0 # Leave 5s for final validation/cleanup
    
    while time.time() < total_deadline:
        # Run SA for a short burst
        burst_time = min(10.0, total_deadline - time.time())
        if burst_time < 1.0:
            break
            
        improved_rows, improved_cost = local_search_sa(best_rows, n_rows, k, rng, time_limit=burst_time)
        
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_rows = improved_rows
        else:
            # If no improvement, we can try to perturb the best solution before restarting
            # Or just break if we've been stuck for a while? 
            # For now, let's just let the loop continue with the same best_rows.
            # To avoid infinite loops of no improvement, we can add a counter.
            pass

    # Final Validation
    # The problem requires valid solutions. If our best solution is invalid (has conflicts),
    # we should try to repair it or return the best valid one found.
    # Our cost function penalizes conflicts, so the best solution *should* be valid if possible.
    # However, if the search ends in an invalid state, we need to ensure we return a valid one.
    # Let's check the best_rows validity.
    
    all_diffs, _ = get_differences(best_rows)
    conflicts = count_conflicts(all_diffs)
    
    if conflicts > 0:
        # Try to repair by greedily fixing conflicts? 
        # Or simply return the greedy init if it was valid and better?
        # The greedy init is always valid by construction.
        init_cost = calculate_cost(initial_rows)
        if init_cost < best_cost:
            best_rows = initial_rows
    
    return best_rows
# EVOLVE-BLOCK-END
