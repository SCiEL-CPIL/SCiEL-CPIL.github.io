import time
import random
import sys

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    m = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    
    # Lower bound on max value M
    # We need total_diffs distinct differences in [1, M].
    # So M must be at least total_diffs.
    # Actually, the differences are a subset of {1, ..., M}.
    # So M >= total_diffs is a necessary condition? No.
    # The differences are distinct integers. The largest difference is at most M.
    # So we need at least total_diffs distinct integers in [1, M].
    # Thus M >= total_diffs.
    lb = total_diffs
    
    # For (4, 7), total_diffs = 4 * 28 = 112.
    # Known best is 134.
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def evaluate(rows):
        """
        Returns (is_valid, score).
        Score is used for minimization.
        If invalid, score is large + penalty.
        If valid, score is max_value.
        """
        used = set()
        max_val = 0
        collisions = 0
        for row in rows:
            # Check structure
            if len(row) != m or row[0] != 0:
                return False, float('inf')
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False, float('inf')
            max_val = max(max_val, row[-1])
            for d in get_diffs(row):
                if d in used:
                    collisions += 1
                else:
                    used.add(d)
        
        if collisions == 0:
            return True, max_val
        else:
            # Penalize collisions heavily and add max_val to guide towards smaller scopes
            # Weight collisions high to prioritize validity
            return False, max_val + 1000 * collisions

    def random_row(rng, max_val):
        """Generate a random increasing row starting with 0, values <= max_val."""
        if max_val < m - 1:
            # Not enough room for m distinct increasing integers starting at 0
            # e.g. 0, 1, ..., m-1 needs max_val >= m-1
            # Just return a dense row, evaluation will fail
            return list(range(m))
        
        # Randomly choose m-1 distinct values from 1..max_val
        # Ensure they are increasing
        # This might be slow if max_val is large and m is small? No, m is small.
        # But we want them to be somewhat spread out or packed?
        # Random sampling is fine.
        # To ensure strictly increasing, we can sample m-1 distinct integers from [1, max_val]
        # and sort them.
        # However, if max_val is close to lb, the space is tight.
        # Let's try to generate a valid-ish row.
        # Greedy with randomization:
        row = [0]
        for _ in range(m - 1):
            # Pick a value > row[-1] and <= max_val
            # Bias towards smaller values to keep scope low?
            # Or uniform?
            # Let's use a mix.
            # Uniform in [row[-1]+1, max_val]
            low = row[-1] + 1
            high = max_val
            if low > high:
                row.append(high) # Should not happen if we checked
            else:
                row.append(rng.randint(low, high))
        
        # Ensure strictly increasing (in case of duplicates, though randint is unique if low<high)
        # Wait, if low == high, it picks that.
        # If we pick 5, then next low is 6.
        # So it is strictly increasing.
        
        # Check if we exceeded max_val? No, we bounded it.
        return row

    def repair_row(row, used_diffs, rng, max_val):
        """
        Attempt to fix a row to be valid with respect to used_diffs and internal conflicts.
        Returns a new row or None if failed.
        """
        # This is a simple local search / backtracking hybrid for a single row
        # Given a set of forbidden diffs, find a valid row.
        
        def backtrack(pos, current_row, current_diffs):
            if pos == m:
                return True, current_row
            
            last_val = current_row[-1]
            # Try values from last_val + 1 to max_val
            # Randomize order?
            # To be efficient, try values that don't conflict.
            
            # MRV-like: just iterate
            # To speed up, we can pre-filter?
            
            candidates = list(range(last_val + 1, max_val + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                ok = True
                new_diffs = set()
                for i in range(pos):
                    d = val - current_row[i]
                    if d in current_diffs or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                
                if ok:
                    success, final_row = backtrack(pos + 1, current_row + [val], current_diffs | new_diffs)
                    if success:
                        return True, final_row
            
            return False, None

        # Start with the given row as a hint? Or just start from scratch?
        # Starting from scratch is safer for validity.
        # But we want to mutate, so maybe start close to original?
        # Let's just start from [0] and backtrack.
        success, new_row = backtrack(1, [0], set())
        return new_row

    def crossover(row1, row2, rng):
        """
        Simple crossover: take elements from row1 or row2 at each position.
        Then repair to ensure increasing and validity?
        Actually, just mixing positions breaks increasing order easily.
        Better: Uniform crossover on the set of values?
        Or: Take a prefix of row1 and suffix of row2?
        Since rows must be increasing, a crossover that preserves order is tricky.
        
        Alternative: 
        1. Generate a new row by taking each element from row1 or row2 with 50% prob.
        2. Sort the resulting list.
        3. Ensure it starts with 0 (force first element to 0).
        4. Ensure strictly increasing (remove duplicates, adjust if needed).
        5. Repair using local search to fix conflicts.
        
        Let's try:
        - Combine values (excluding 0s)
        - Shuffle
        - Pick m-1 smallest? No, that biases low.
        - Pick m-1 random from union?
        
        Let's stick to a simpler mutation-heavy approach if crossover is hard.
        But the directive asks for crossover.
        
        Let's try "Order-based crossover":
        - Take a random subset of positions from row1.
        - Fill remaining positions with values from row2 that are not used, maintaining order?
        This is complex.
        
        Simpler Crossover:
        - Create a new row by selecting each element independently from row1 or row2.
        - Sort the selection.
        - If length < m, add random small integers.
        - If length > m, keep smallest m?
        - Then repair.
        """
        # Get non-zero values
        v1 = row1[1:]
        v2 = row2[1:]
        
        # Combine
        pool = []
        for i in range(m - 1):
            if rng.random() < 0.5:
                pool.append(v1[i])
            else:
                pool.append(v2[i])
        
        # Sort
        pool.sort()
        
        # Ensure strictly increasing and start with 0
        new_row = [0]
        last = 0
        for v in pool:
            if v > last:
                new_row.append(v)
                last = v
            if len(new_row) == m:
                break
        
        # If we didn't get enough values, fill with random small increments
        while len(new_row) < m:
            new_row.append(last + 1)
            last += 1
            
        # If we have more than m (shouldn't happen due to break, but just in case)
        # And ensure max value is not too large?
        
        return new_row

    def mutate(row, used_diffs, rng, max_val):
        """
        Local search mutation:
        1. Identify conflicts.
        2. Shift elements to resolve.
        """
        # Calculate current conflicts
        internal_diffs = get_diffs(row)
        conflicts_internal = 0
        # Check internal duplicates
        # get_diffs returns a set, so we lose info on duplicates.
        # Let's count properly.
        all_diffs = []
        for i in range(m):
            for j in range(i + 1, m):
                all_diffs.append(row[j] - row[i])
        
        # Count duplicates
        from collections import Counter
        counts = Counter(all_diffs)
        internal_conflicts = sum(c - 1 for c in counts.values() if c > 1)
        
        external_conflicts = 0
        for d in all_diffs:
            if d in used_diffs:
                external_conflicts += 1
        
        total_conflicts = internal_conflicts + external_conflicts
        
        if total_conflicts == 0:
            return row
        
        # Try to fix by shifting a random element
        # Pick a random position (1 to m-1)
        pos = rng.randint(1, m - 1)
        
        # Determine range for new value
        # Must be > row[pos-1] and < row[pos+1] (if exists)
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < m - 1 else max_val
        
        if low > high:
            # Can't shift much, try a different position or just random jump
            # Random jump within [row[pos-1]+1, max_val]
            # But must maintain order.
            # If we jump too high, we might break order with next element.
            # So we must also adjust subsequent elements?
            # This is getting complicated.
            
            # Alternative: Re-generate the suffix from pos onwards using repair_row logic?
            # Let's try a simpler mutation: Randomly change one value to a random valid value
            # and then "push" subsequent values up if necessary to maintain order.
            
            new_val = rng.randint(low, max_val)
            # If new_val >= row[pos+1], we need to increase row[pos+1] etc.
            # This can cascade.
            
            # Let's implement a "push" mutation
            new_row = row[:]
            new_row[pos] = new_val
            
            # Fix order
            for i in range(pos + 1, m):
                if new_row[i] <= new_row[i - 1]:
                    new_row[i] = new_row[i - 1] + 1
                    
            # Check if we exceeded max_val
            if new_row[-1] > max_val:
                # Failed, return original or try to compress?
                # Just return original to avoid invalid scope?
                # Or return a compressed version?
                # Let's return original.
                return row
            
            return new_row

    # Initialize population
    pop_size = 50
    population = []
    
    # First, get a good upper bound using greedy
    def greedy_construct(rng):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(m - 1):
                x = row[-1] + 1
                while True:
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
                    x += 1
            rows.append(row)
        return rows

    # Start with a greedy solution to set the initial max_val
    initial_rows = greedy_construct(random.Random(42))
    initial_max = max(max(r) for r in initial_rows)
    
    # We will search in the range [lb, initial_max]
    # But GA works better with a fixed bound or adaptive.
    # Let's use adaptive max_val.
    
    # Initialize population with variations of the greedy solution and random rows
    for i in range(pop_size):
        if i == 0:
            row = initial_rows[0][:]
        elif i == 1:
            row = initial_rows[1][:]
        else:
            # Random row within [0, initial_max]
            row = random_row(random.Random(i), initial_max)
        # Ensure validity of structure (increasing)
        # random_row does this.
        population.append(row)
        
    # Wait, the "individual" is a SET of rows.
    # So population should be a list of "solutions", where each solution is a list of n_rows rows.
    
    population = []
    for i in range(pop_size):
        if i == 0:
            sol = [r[:] for r in initial_rows]
        else:
            sol = []
            for _ in range(n_rows):
                sol.append(random_row(random.Random(i * 100 + len(sol)), initial_max))
        population.append(sol)

    best_solution = None
    best_score = float('inf')
    best_valid = False

    def evaluate_solution(sol):
        used = set()
        max_val = 0
        collisions = 0
        for row in sol:
            if len(row) != m or row[0] != 0:
                return False, float('inf'), 0
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False, float('inf'), 0
            max_val = max(max_val, row[-1])
            for d in get_diffs(row):
                if d in used:
                    collisions += 1
                else:
                    used.add(d)
        if collisions == 0:
            return True, max_val, max_val
        else:
            return False, max_val + 1000 * collisions, max_val + 1000 * collisions

    # Evaluate initial population
    for sol in population:
        valid, score, _ = evaluate_solution(sol)
        if score < best_score:
            best_score = score
            best_solution = [r[:] for r in sol]
            best_valid = valid

    # GA Loop
    generations = 0
    while time.time() < deadline - 5:
        generations += 1
        
        # Selection: Tournament selection
        def select_tournament(pop, k=3):
            candidates = random.sample(pop, k)
            best_cand = min(candidates, key=lambda s: evaluate_solution(s)[1])
            return best_cand

        new_population = []
        
        # Elitism: Keep the best solution
        new_population.append([r[:] for r in best_solution])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
                
            p1 = select_tournament(population)
            p2 = select_tournament(population)
            
            # Crossover
            if random.random() < 0.8:
                child = []
                for r in range(n_rows):
                    # Crossover rows
                    # Use the crossover function defined earlier
                    # But we need to pass used_diffs? No, crossover is structural.
                    # The repair will handle conflicts.
                    child_row = crossover(p1[r], p2[r], random)
                    child.append(child_row)
            else:
                # Just copy parent
                child = [r[:] for r in p1]
            
            # Mutation
            # For each row in child, apply mutation
            # We need to know the "used_diffs" from other rows to guide mutation?
            # For simplicity, mutate each row independently, then evaluate the whole set.
            # Or, sequential mutation: mutate row 0, then row 1 given row 0, etc.
            
            # Let's do sequential repair/mutation to be smarter.
            # This is expensive but effective.
            
            # Actually, let's just do random mutation on values.
            for r_idx in range(n_rows):
                if random.random() < 0.7: # Mutation rate
                    # Pick a random element to shift
                    pos = random.randint(1, m - 1)
                    # Shift by small amount
                    delta = random.choice([-2, -1, 1, 2, 3])
                    old_val = child[r_idx][pos]
                    new_val = old_val + delta
                    
                    # Check bounds and order
                    if new_val > child[r_idx][pos - 1] and (pos == m - 1 or new_val < child[r_idx][pos + 1]):
                        child[r_idx][pos] = new_val
                    else:
                        # If order broken, try to fix by adjusting neighbors?
                        # Or just reject mutation for this element
                        pass
            
            # Check if child is valid and better
            valid, score, _ = evaluate_solution(child)
            if score < best_score:
                best_score = score
                best_solution = [r[:] for r in child]
                best_valid = valid
            
            new_population.append(child)
        
        population = new_population
        
        # If we found a valid solution, we can try to lower the max_val bound?
        # The GA doesn't strictly enforce a max_val, but the fitness penalizes large max_val.
        # So it will naturally search for smaller max_vals if valid solutions exist.

    return best_solution

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
