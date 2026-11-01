"""Difference triangle set (4,7): construct 4 rows of 8 increasing integers starting at 0
with all pairwise differences distinct across all rows, minimizing the largest entry."""
import time
import random
import sys

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, minimizing the largest entry."""
    deadline = time.time() + 115  # safety margin under 120s wall-clock
    
    # For this task: n_rows=4, k=7, so each row has 8 entries.
    # Total differences per row: C(8,2) = 28. Total across 4 rows: 112 differences.
    # The differences must all be distinct positive integers.
    
    def get_differences(rows):
        """Return the set of all differences across all rows."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def count_conflicts(rows):
        """Count the number of duplicate differences (should be 0 for valid)."""
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return len(diffs) - len(set(diffs))
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        return count_conflicts(rows) == 0
    
    def greedy_build(seed=0):
        """Build rows greedily, picking smallest valid next element."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
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
    
    def greedy_interleaved(seed=0):
        """Build rows interleaved: add one element to each row in round-robin fashion."""
        rng = random.Random(seed)
        used = set()
        rows = [[0] for _ in range(n_rows)]
        
        while any(len(row) < k + 1 for row in rows):
            for r in range(n_rows):
                if len(rows[r]) >= k + 1:
                    continue
                x = max(rows[r]) + 1
                # Try to find the smallest valid x
                found = False
                while not found:
                    new_diffs = {x - a for a in rows[r]}
                    if not new_diffs & used:
                        rows[r].append(x)
                        used |= new_diffs
                        found = True
                    else:
                        x += 1
        return rows
    
    def greedy_randomized(seed=0):
        """Build rows with some randomness in the choice of next element."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Sometimes skip ahead randomly
                if rng.random() < 0.2:
                    x += rng.randint(1, 5)
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def local_search(rows, time_limit):
        """Iterative local search to reduce the maximum entry."""
        rng = random.Random()
        best_rows = [list(r) for r in rows]
        best_max = max_entry(best_rows)
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            rows_copy = [list(r) for r in best_rows]
            improved = False
            
            # Try moving one element in a random row to a smaller value
            for _ in range(100):
                r = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, k)
                old_val = rows_copy[r][pos]
                
                lo = rows_copy[r][pos - 1] + 1
                hi = old_val - 1 if pos < k else old_val
                
                if lo > hi:
                    continue
                    
                # Try values from old_val-1 down to lo
                for new_val in range(old_val - 1, lo - 1, -1):
                    if pos < k and new_val >= rows_copy[r][pos + 1]:
                        continue
                    if pos > 1 and new_val <= rows_copy[r][pos - 1]:
                        continue
                    
                    rows_copy[r][pos] = new_val
                    if is_valid(rows_copy):
                        cur_max = max_entry(rows_copy)
                        if cur_max < best_max:
                            best_max = cur_max
                            best_rows = [list(rr) for rr in rows_copy]
                            improved = True
                        break
                    rows_copy[r][pos] = old_val
                
                if improved:
                    break
            
            # If no improvement, try rebuilding one row
            if not improved:
                r = rng.randint(0, n_rows - 1)
                other_rows = [best_rows[i] for i in range(n_rows) if i != r]
                
                used = set()
                for row in other_rows:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            used.add(row[j] - row[i])
                
                for attempt in range(30):
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        x += 1
                        if rng.random() < 0.3:
                            x += rng.randint(1, 3)
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                    
                    if max(row) < best_max:
                        new_rows = [best_rows[i] for i in range(n_rows)]
                        new_rows[r] = row
                        if is_valid(new_rows):
                            best_max = max_entry(new_rows)
                            best_rows = new_rows
                            improved = True
                            break
            
            if time.time() > deadline - 5:
                break
        
        return best_rows, best_max
    
    def simulated_annealing(rows, time_limit):
        """Simulated annealing to escape local optima."""
        rng = random.Random()
        current_rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_cost = max_entry(best_rows) * 1000 + count_conflicts(best_rows)
        current_cost = best_cost
        start_time = time.time()
        temperature = 50.0
        cooling = 0.995
        
        while time.time() - start_time < time_limit:
            # Generate a neighbor
            rows_copy = [list(r) for r in current_rows]
            r = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)
            old_val = rows_copy[r][pos]
            
            # Try to move the element
            direction = rng.choice([-1, 1])
            step = rng.randint(1, 10)
            new_val = old_val + direction * step
            
            # Ensure validity of position
            if pos > 0 and new_val <= rows_copy[r][pos - 1]:
                new_val = rows_copy[r][pos - 1] + 1
            if pos < k and new_val >= rows_copy[r][pos + 1]:
                new_val = rows_copy[r][pos + 1] - 1
            
            if new_val <= 0 or (pos < k and new_val >= rows_copy[r][pos + 1]):
                continue
            
            rows_copy[r][pos] = new_val
            new_cost = max_entry(rows_copy) * 1000 + count_conflicts(rows_copy)
            
            # Accept or reject
            if new_cost < current_cost:
                current_rows = rows_copy
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows = [list(r) for r in current_rows]
            else:
                # Metropolis criterion
                if temperature > 0.01:
                    delta = new_cost - current_cost
                    if delta < 1000:  # Only accept if conflict increase is small
                        prob = 1.0 / (1.0 + delta / max(temperature, 0.01))
                        if rng.random() < prob:
                            current_rows = rows_copy
                            current_cost = new_cost
            
            temperature *= cooling
        
        return best_rows, best_cost
    
    # Phase 1: Generate multiple initial solutions
    initial_solutions = []
    for seed in [42, 123, 456, 789, 1000, 2024]:
        sol = greedy_build(seed)
        initial_solutions.append(sol)
        sol = greedy_interleaved(seed)
        initial_solutions.append(sol)
        sol = greedy_randomized(seed)
        initial_solutions.append(sol)
    
    # Find the best initial solution
    best_initial = min(initial_solutions, key=max_entry)
    best_max = max_entry(best_initial)
    
    # Phase 2: Local search on the best initial solution
    remaining_time = deadline - time.time() - 20
    if remaining_time > 5:
        best_rows, best_max = local_search(best_initial, remaining_time)
    
    # Phase 3: Simulated annealing for further improvement
    remaining_time = deadline - time.time() - 10
    if remaining_time > 5:
        sa_rows, sa_cost = simulated_annealing(best_rows, remaining_time)
        if sa_cost // 1000 < best_max:
            best_rows = sa_rows
            best_max = sa_cost // 1000
    
    # Phase 4: Final local polish
    remaining_time = deadline - time.time() - 3
    if remaining_time > 2:
        best_rows, best_max = local_search(best_rows, remaining_time)
    
    return best_rows

# Quick self-test
if __name__ == "__main__":
    rows = construct(4, 7)
    print("Rows:", rows)
    print("Max entry:", max(max(r) for r in rows))
    # Verify validity
    diffs = set()
    valid = True
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    print(f"Duplicate difference {d}")
                diffs.add(d)
    print("Valid:", valid)
    print("Num differences:", len(diffs))
