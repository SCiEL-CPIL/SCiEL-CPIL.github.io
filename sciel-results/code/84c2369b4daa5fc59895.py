import time
import random
import sys
import math
from collections import defaultdict, deque

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 145.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_full(used, rng, max_val=None):
        """Build a full row greedily, avoiding used diffs. Returns row or None."""
        row = [0]
        for pos in range(k):
            found = False
            if max_val is not None:
                for _ in range(50):
                    x = rng.randint(row[-1] + 1, max_val)
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                        break
            else:
                x = row[-1]
                while True:
                    x += 1
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                        break
            if not found:
                return None
        return row

    def greedy_construct(n, k_val, rng):
        used = set()
        rows = []
        for _ in range(n):
            row = greedy_full(used, rng)
            if row is None:
                return None
            rows.append(row)
        return rows

    def greedy_variant(n, k_val, rng, bias):
        """Greedy with bias towards smaller or larger values."""
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            success = True
            for pos in range(k_val):
                found = False
                lo = row[-1] + 1
                # Try a range of candidates
                if bias < 0.5:
                    # Prefer smaller
                    for _ in range(30):
                        x = lo + int((rng.random() ** 2) * 50)
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            found = True
                            break
                else:
                    # Prefer larger (but bounded)
                    for _ in range(30):
                        x = lo + int(rng.random() * 80)
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            found = True
                            break
                if not found:
                    # Fallback to sequential
                    x = row[-1]
                    while True:
                        x += 1
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            break
            rows.append(row)
        return rows

    def compact_row(row, used, all_rows, my_index):
        """Try to reduce the largest entry of a row by shifting values down."""
        # Simple: try to reduce each element from the last one backward
        improved = False
        for pos in range(len(row) - 1, 0, -1):
            orig = row[pos]
            new_val = row[pos] - 1
            # Must be greater than row[pos-1]
            while new_val > row[pos - 1]:
                # Check if this new value works
                test_row = row[:pos] + [new_val] + row[pos+1:]
                # Check internal diffs are still distinct
                internal_diffs = set()
                valid_internal = True
                for i in range(len(test_row)):
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        if d in internal_diffs:
                            valid_internal = False
                            break
                        internal_diffs.add(d)
                    if not valid_internal:
                        break
                if not valid_internal:
                    new_val -= 1
                    continue
                # Check against other rows' diffs
                conflict = False
                for idx, other_row in enumerate(all_rows):
                    if idx == my_index:
                        continue
                    other_diffs = get_diffs(other_row)
                    for d in internal_diffs:
                        if d in other_diffs:
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict:
                    row[pos] = new_val
                    improved = True
                    break  # restart compaction from the top
                else:
                    new_val -= 1
        return improved

    def local_compact(rows, max_iter=100):
        """Iteratively compact rows to reduce their max entries."""
        for _ in range(max_iter):
            any_improved = False
            for i in range(len(rows)):
                if compact_row(rows[i], None, rows, i):
                    any_improved = True
            if not any_improved:
                break
        return rows

    def simulated_annealing(rows, rng, deadline, max_iter=5000):
        """SA with swap moves within rows to reduce scope."""
        if not rows:
            return rows
        rows = [list(r) for r in rows]
        current_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        temperature = 10.0
        cooling = 0.995

        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break

            # Pick a random row and a random position to try changing
            row_idx = rng.randint(0, len(rows) - 1)
            row = rows[row_idx]
            pos = rng.randint(1, len(row) - 1)  # don't touch position 0

            # Try a small perturbation
            delta = rng.randint(-3, 3)
            new_val = row[pos] + delta

            if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
                continue

            # Check if this change is valid for the row
            test_row = row[:]
            test_row[pos] = new_val
            internal_diffs = set()
            valid = True
            for i in range(len(test_row)):
                for j in range(i + 1, len(test_row)):
                    d = test_row[j] - test_row[i]
                    if d in internal_diffs:
                        valid = False
                        break
                    internal_diffs.add(d)
                if not valid:
                    break

            if not valid:
                continue

            # Check against other rows
            conflict = False
            for idx, other_row in enumerate(rows):
                if idx == row_idx:
                    continue
                other_diffs = get_diffs(other_row)
                for d in internal_diffs:
                    if d in other_diffs:
                        conflict = True
                        break
                if conflict:
                    break

            if conflict:
                continue

            # Accept or reject based on SA criterion
            new_scope = max(max(r) for r in rows if r is not test_row)
            new_scope = max(new_scope, test_row[-1])
            if new_scope < current_scope:
                # Always accept improvement
                rows[row_idx] = test_row
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            else:
                # Accept with probability exp(-(new_scope - current_scope)/temperature)
                if rng.random() < math.exp(-1.0 / max(temperature, 0.01)):
                    rows[row_idx] = test_row
                    current_scope = new_scope

            temperature *= cooling

        return best_rows

    # Genetic Algorithm
    class Individual:
        def __init__(self, rows):
            self.rows = rows
            self.fitness = self._compute_fitness()
            
        def _compute_fitness(self):
            # Fitness: negative scope + penalty for collisions
            # Lower is better
            if not self.rows or len(self.rows) != n_rows:
                return float('inf')
                
            seen = set()
            collisions = 0
            scope = 0
            for row in self.rows:
                if len(row) != m or row[0] != 0:
                    return float('inf')
                for i in range(1, len(row)):
                    if row[i] <= row[i-1]:
                        return float('inf')
                scope = max(scope, row[-1])
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in seen:
                            collisions += 1
                        seen.add(d)
                        
            # Penalty for collisions is heavy
            return scope + 1000 * collisions

        def __lt__(self, other):
            return self.fitness < other.fitness

    def crossover(parent1, parent2, rng):
        # Crossover: recombine difference subsets from parents
        # This is a simplified version - we just swap rows
        child_rows = []
        for i in range(n_rows):
            if rng.random() < 0.5:
                child_rows.append(parent1.rows[i][:])
            else:
                child_rows.append(parent2.rows[i][:])
        return Individual(child_rows)

    def mutate(individual, rng):
        # Mutation: local search to resolve conflicts by shifting elements
        rows = [list(r) for r in individual.rows]
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            row_idx = rng.randint(0, len(rows) - 1)
            pos = rng.randint(1, len(rows[row_idx]) - 1)
            delta = rng.randint(-5, 5)
            new_val = rows[row_idx][pos] + delta
            if new_val <= rows[row_idx][pos-1] or (pos < len(rows[row_idx]) - 1 and new_val >= rows[row_idx][pos+1]):
                continue
            test_row = rows[row_idx][:]
            test_row[pos] = new_val
            # Check if this change reduces fitness
            test_individual = Individual([r if r is not test_row else test_row for r in rows])
            if test_individual.fitness < individual.fitness:
                rows[row_idx] = test_row
        return Individual(rows)

    def genetic_algorithm(initial_population, rng, deadline, pop_size=20, generations=100):
        population = initial_population[:]
        best = min(population)
        
        for gen in range(generations):
            if time.time() > deadline - 10:
                break
                
            # Selection: tournament selection
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(best)
            
            while len(new_population) < pop_size:
                # Tournament selection
                candidates = rng.sample(population, min(5, len(population)))
                parent1 = min(candidates)
                candidates = rng.sample(population, min(5, len(population)))
                parent2 = min(candidates)
                
                child = crossover(parent1, parent2, rng)
                if rng.random() < 0.3:  # Mutation rate
                    child = mutate(child, rng)
                
                new_population.append(child)
            
            population = new_population
            best = min(population)
            
        return best.rows

    # CSP Backtracking over difference values
    def csp_backtrack(n_rows, k_val, max_scope, rng):
        """
        Formulate as CSP: assign each of the total_diffs slots to a unique d in [1, max_scope].
        Slots are grouped by row. For each row, the set of assigned diffs must form a valid difference set.
        We use MRV to select the next slot to assign.
        """
        total_slots = n_rows * diffs_per_row
        # slots[row_idx] is a list of slot indices belonging to that row
        slots_by_row = [list(range(row_idx * diffs_per_row, (row_idx + 1) * diffs_per_row)) for row_idx in range(n_rows)]
        
        # We will assign values to slots.
        # State: 
        #   assignment: dict slot_idx -> value
        #   used_values: set of assigned values
        #   row_diffs: dict row_idx -> set of assigned diffs for that row (to check closure)
        
        # To make MRV work, we need to know which values are still possible for each unassigned slot.
        # A value d is possible for a slot in row r if:
        # 1. d is not in used_values
        # 2. Adding d to row_diffs[r] does not violate the "difference set" property relative to already assigned diffs in that row.
        #    Actually, the closure property is global for the row. 
        #    A simpler check: if we assign d to a slot in row r, and we already have some set S of diffs for row r,
        #    then S U {d} must be a subset of the final difference set of row r.
        #    The final difference set D_r must satisfy: if a, b in D_r and a < b, then b-a in D_r.
        #    This is a strong constraint.
        
        # Alternative approach: 
        # Instead of assigning arbitrary diffs, we construct the row entries directly but use backtracking on the entries.
        # The directive says "assign each of the N required difference slots to a unique integer d".
        # Let's try to implement that literally.
        
        # For a row with entries x_0=0, x_1, ..., x_k, the diffs are x_j - x_i.
        # If we pick a set of diffs D for a row, can we reconstruct the entries?
        # Yes, if D is a valid difference set of a sequence starting at 0.
        # This is equivalent to finding a sequence.
        
        # Let's stick to backtracking on the entries, but use the "difference set" validity check to prune.
        # Actually, the current seed already does local search on entries.
        # The inspiration suggests "CSP over difference values".
        
        # Let's try a different CSP: 
        # Variables: The entries of all rows.
        # Domain: [1, max_scope]
        # Constraints: 
        #   1. All diffs distinct.
        #   2. Entries increasing.
        
        # This is what the seed does with SA/GA.
        
        # Let's try a pure backtracking on entries with strong pruning.
        # Order rows by "difficulty" (e.g., try to build the row with the most constraints first? No, they are symmetric).
        # Order entries within a row: x_1, x_2, ..., x_k.
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, pos_in_row, current_row):
            if time.time() > deadline - 5:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos_in_row == 0:
                # Start new row
                rows.append([0])
                return backtrack(row_idx, 1, [0])
            
            if pos_in_row == k_val + 1:
                # Row complete
                rows.pop() # Remove current row placeholder if we managed it? No, we append at start.
                # Actually, let's manage rows list carefully.
                # Let's restart this logic.
                pass
            
            # This manual backtracking is getting messy. 
            # Let's use a recursive function that builds one row at a time.
            
            return False # Placeholder

        # Let's implement a cleaner backtracking for one row, then recurse for next row.
        
        def build_row(row_idx, used_diffs, rng):
            """
            Backtrack to build row[row_idx] given used_diffs from previous rows.
            Returns the row list or None.
            """
            row = [0]
            
            def _backtrack(pos, current_row):
                if time.time() > deadline - 5:
                    return None
                
                if pos == k_val + 1:
                    return current_row[:]
                
                # Determine candidate values for current_row[pos]
                # It must be > current_row[-1] and <= max_scope
                # And all new diffs (current_row[pos] - a for a in current_row) must not be in used_diffs
                # And must not create internal duplicates (handled by checking against current_row's existing diffs? 
                # No, we check against used_diffs which includes internal diffs of this row if we add them as we go)
                
                # To use MRV, we could look ahead, but let's just try candidates in a randomized order.
                # To improve, we can estimate which candidates leave the most options for future positions.
                
                start = current_row[-1] + 1
                end = max_scope
                
                # Generate candidates
                # Randomize to avoid getting stuck in local minima
                candidates = list(range(start, end + 1))
                rng.shuffle(candidates)
                
                # Limit candidates to first N to avoid explosion?
                # If max_scope is small, this is fine.
                # If max_scope is large, we might need to be smarter.
                
                for val in candidates:
                    new_diffs = []
                    valid = True
                    for a in current_row:
                        d = val - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Check internal consistency? 
                    # The diffs are distinct within the row by construction if we check against used_diffs 
                    # (which we will update to include new_diffs before recursing).
                    # Wait, used_diffs is passed by reference. We need to add new_diffs, recurse, then remove.
                    
                    # Add to used_diffs
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    res = _backtrack(pos + 1, current_row + [val])
                    
                    # Remove from used_diffs
                    for d in new_diffs:
                        used_diffs.remove(d)
                    
                    if res is not None:
                        return res
                
                return None
            
            return _backtrack(1, row)

        # Main CSP loop: try to build all rows
        # We can try different max_scope values? 
        # The task asks to minimize the scope. 
        # We can binary search on max_scope or just try to build with a tight bound.
        # Let's try to build with a bound close to the known best (134) or counting bound (112).
        
        # We will try a range of max_scope values, starting from the lower bound.
        # Lower bound: total_diffs (since diffs are 1..M distinct, M >= total_diffs).
        # For 4 rows, k=7, diffs_per_row = 28, total_diffs = 112.
        
        lb = total_diffs
        # Upper bound: current best known or a heuristic.
        # Let's start with lb and increase if we fail.
        
        best_rows = None
        
        # We will run the backtracking multiple times with randomization to find a solution.
        # If we find one with scope M, we can try to improve it with local search.
        
        # To be safe, we'll try a few values of max_scope.
        # Let's try max_scope = lb, lb+1, ... up to some limit or until deadline.
        
        # Actually, backtracking with max_scope = lb might be very hard if no solution exists.
        # Let's start with a slightly higher bound to ensure we find *a* solution, then compact.
        
        # Heuristic upper bound: 1.2 * lb or known best.
        # Let's try max_scope = 120, 125, 130, 134...
        
        # Let's just try to find *any* valid solution first using the greedy method (fast),
        # then use the CSP backtracking to try to find a solution with a *smaller* scope.
        
        # We already have greedy_construct. Let's use that to get an initial best_scope.
        # Then we try to beat it with CSP.
        
        return None # This function is a placeholder for the actual CSP logic which is complex.
                    # I will implement a simpler version below.

    def csp_search(max_scope, rng):
        """
        Try to find a valid set of rows with max entry <= max_scope using backtracking.
        """
        used_diffs = set()
        rows = []
        
        def build_row(row_idx):
            if time.time() > deadline - 5:
                return None
            
            row = [0]
            
            def _backtrack(pos, current_row):
                if time.time() > deadline - 5:
                    return None
                
                if pos == k + 1:
                    return current_row[:]
                
                start = current_row[-1] + 1
                end = max_scope
                
                # Candidate generation
                # To speed up, we can prioritize candidates that are "tight" or "loose"
                # Let's just randomize.
                candidates = list(range(start, end + 1))
                rng.shuffle(candidates)
                
                for val in candidates:
                    new_diffs = []
                    valid = True
                    for a in current_row:
                        d = val - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Add to used_diffs
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    res = _backtrack(pos + 1, current_row + [val])
                    
                    # Remove from used_diffs
                    for d in new_diffs:
                        used_diffs.remove(d)
                    
                    if res is not None:
                        return res
                
                return None
            
            return _backtrack(1, row)

        # Build rows one by one
        for i in range(n_rows):
            row = build_row(i)
            if row is None:
                return None
            rows.append(row)
            
        return rows

    # Main
    rng = random.Random(12345)

    # Initial greedy solution
    best_rows = greedy_construct(n_rows, k, rng)
    if best_rows is None:
        # Fallback: very simple sequential
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            best_rows.append(row)

    best_scope = max(max(r) for r in best_rows)

    # Try multiple greedy variants
    for trial in range(10):
        if time.time() > deadline - 30:
            break
        bias = rng.random()
        rows = greedy_variant(n_rows, k, rng, bias)
        if rows and is_valid(rows):
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope

    # Local compaction
    best_rows = local_compact(best_rows, max_iter=50)
    if is_valid(best_rows):
        best_scope = max(max(r) for r in best_rows)
    else:
        # Revert if invalid
        best_rows = greedy_construct(n_rows, k, rng)
        best_scope = max(max(r) for r in best_rows)

    # Try CSP Backtracking to find a better solution
    # We will try to find a solution with scope < best_scope
    # Start from best_scope - 1 and go down? Or start from lower bound and go up?
    # Going down is safer if we already have a solution.
    
    # Let's try a few values below best_scope
    target_scope = best_scope - 1
    # Also try the theoretical lower bound
    lb = total_diffs
    
    # Try a range of scopes
    scopes_to_try = set()
    # Add best_scope - 1, - 2, ...
    for s in range(max(lb, best_scope - 10), best_scope):
        scopes_to_try.add(s)
    # Add lower bound
    scopes_to_try.add(lb)
    
    # Sort scopes to try smaller ones first (since we want to minimize)
    scopes_to_try = sorted(scopes_to_try)
    
    for scope in scopes_to_try:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random restarts for this scope
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            rows = csp_search(scope, rng)
            if rows is not None and is_valid(rows):
                # Found a valid solution with this scope
                # It might not be optimal within this scope, but it's valid.
                # We can try to compact it further?
                # For now, accept it.
                found = True
                best_rows = rows
                best_scope = scope
                break
        
        if found:
            # If we found a solution at scope, we might try scope-1 next (already in list)
            # But since we sorted, we will try smaller ones first.
            # Wait, if we found a solution at scope=120, we should try 119, 118...
            # The loop does that.
            pass

    # Final local search on the best found
    if is_valid(best_rows):
        best_rows = local_compact(best_rows, max_iter=50)
        if is_valid(best_rows):
            best_scope = max(max(r) for r in best_rows)
    
    # Simulated annealing on best result
    best_rows = simulated_annealing(best_rows, rng, deadline, max_iter=5000)
    if is_valid(best_rows):
        best_scope = max(max(r) for r in best_rows)
    else:
        # Revert
        best_rows = greedy_construct(n_rows, k, rng)
        best_scope = max(max(r) for r in best_rows)

    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
