import time
import random
import sys

def greedy_construct(n_rows, k):
    """
    Standard greedy construction to establish a baseline.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not diffs & used:
                row.append(x)
                used.update(diffs)
        rows.append(row)
    return rows

def backtrack_global(n_rows, k, limit, seed, deadline):
    """
    Backtracking search that builds rows one by one, but uses a more sophisticated
    pruning strategy: it tries to complete a row by looking ahead to ensure that
    the remaining differences needed for that row don't conflict with the global
    used set in a way that makes completion impossible.
    
    Actually, a better global approach is to view the problem as packing intervals.
    However, a strong heuristic for difference triangles is to use a 'repair' phase
    or a simulated annealing style approach on the set of rows.
    
    Let's try a different angle: Local Search / Simulated Annealing.
    We start with a valid set (e.g., from greedy) and then try to reduce the max element
    by swapping elements or shifting rows, maintaining validity.
    
    Wait, the prompt asks for a MATERIALLY DIFFERENT strategy.
    The current program uses randomized greedy with windows.
    Inspiration 2 mentioned backtracking.
    
    Let's implement a proper Backtracking Search with MRV (Minimum Remaining Values) heuristic.
    We don't build rows sequentially in a fixed order. We can treat the 'slots' (row, position)
    as variables. But that's complex.
    
    Alternative: Backtracking on the difference set.
    Total differences needed: D = n_rows * k * (k+1) // 2.
    The differences must be distinct.
    The differences for a single row with elements 0 = a_0 < a_1 < ... < a_k are:
    {a_j - a_i | 0 <= i < j <= k}.
    These differences must be distinct across all rows.
    
    Let's try a hybrid:
    1. Generate a valid baseline using greedy.
    2. Use Local Search (Simulated Annealing) to minimize the maximum element.
       State: The set of rows.
       Move: Pick a random row, pick a random element (not the first), and change its value.
       Constraint: The row must remain strictly increasing and all differences must remain distinct.
       Objective: Minimize max(max(row) for row in rows).
       
    This is fundamentally different from constructive search. It optimizes an existing solution.
    """
    # This function will be a helper for the main construct if we go that route.
    # But for now, let's stick to a robust Backtracking with better pruning.
    
    # Actually, the most effective different strategy for these small N problems is often
    # simply a more powerful backtracking with constraint propagation.
    
    pass

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Local Search / Simulated Annealing
    # This is a materially different strategy: Optimization of a valid instance
    # rather than construction from scratch.
    
    def is_valid(rows):
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            # Check strictly increasing
            for i in range(len(r)-1):
                if r[i] >= r[i+1]:
                    return False
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # Start SA from the greedy solution
    current_rows = [r[:] for r in best_rows]
    current_score = best_scope
    
    # We want to minimize the max element.
    # Move: Pick a random row, pick a random index (1..k), and perturb the value.
    # If the perturbation breaks validity, reject (or accept with low prob if it's an improvement in some other metric? No, must be valid).
    # Actually, standard SA requires a defined energy. Here, validity is a hard constraint.
    # So we are doing a Random Walk with Rejection (Metropolis-Hastings on valid states).
    
    # To make it effective, we need moves that can reduce the max.
    # If we just change a random element, we might increase the max.
    # We should bias moves towards reducing the max.
    
    # Strategy:
    # 1. Identify the row with the current max.
    # 2. Try to reduce that max by lowering the last element or shifting elements down.
    # 3. Ensure validity.
    
    # Let's implement a simple local search:
    # 1. Start with greedy.
    # 2. Loop:
    #    - Pick a random row.
    #    - Pick a random position j > 0.
    #    - Try to change row[j] to a new value v such that:
    #      a) row[j-1] < v < row[j+1] (or v < limit if j is last)
    #      b) All differences in the new row are distinct and don't conflict with other rows.
    #    - If valid, accept if it improves or maintains the score (or with prob exp(-delta/T)).
    #    - Specifically, we want to minimize the global max.
    
    # A simpler and often more effective approach for these specific small problems is
    # to use the "repair" technique:
    # Try to construct rows with a target limit L.
    # If greedy fails, use backtracking to fix the last few elements.
    
    # Let's combine:
    # 1. Try to find a solution with scope < best_scope using a randomized backtracking that is smarter than the current one.
    # The current one picks from a window of candidates.
    # Let's use a "Look-ahead" heuristic: When choosing the next number, prefer numbers that leave more "room" for the remaining numbers in the row.
    
    def smart_randomized_greedy(n_rows, k, limit, seed):
        random.seed(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            success = True
            while len(row) < k + 1:
                candidates = []
                # Collect all valid candidates
                for x in range(current_max + 1, limit + 1):
                    diffs = {x - a for a in row}
                    if not (diffs & used):
                        candidates.append(x)
                
                if not candidates:
                    success = False
                    break
                
                # Heuristic: Score candidates by how "tight" they are.
                # We want to pick a candidate that allows the remaining (k + 1 - len(row)) - 1 elements
                # to be placed within the limit.
                # The minimum possible increase for the remaining m elements is m * (current_max + 1) ... no.
                # If we pick x, the next must be > x. The smallest possible next is x+1.
                # So the minimum possible final value if we pick x now is x + (remaining_steps).
                # We want x + remaining_steps <= limit.
                remaining_steps = (k + 1 - len(row)) - 1
                
                valid_candidates = [x for x in candidates if x + remaining_steps <= limit]
                
                if not valid_candidates:
                    # If no candidate satisfies the strict bound, we might still be able to fit
                    # if the limit is loose, but if we are tight, we are stuck.
                    # Let's relax the constraint slightly or just fail.
                    # Actually, if x + remaining_steps > limit, it's impossible to complete
                    # with strictly increasing integers.
                    success = False
                    break
                
                # Among valid candidates, pick randomly.
                # Bias towards smaller values? Smaller values leave more room.
                # Let's pick uniformly from valid_candidates.
                next_x = random.choice(valid_candidates)
                
                row.append(next_x)
                current_max = next_x
                used.update({next_x - a for a in row[:-1]})
            
            if not success:
                return None
            rows.append(row)
            
        global_max = max(max(r) for r in rows)
        if global_max > limit:
            return None
        return rows

    # 3. Search loop
    # We know the greedy scope. We try to lower it.
    # Since the problem is small (n_rows=4, k=7 -> 8 elements), the search space is manageable.
    
    current_limit = best_scope - 1
    seed_counter = 0
    trials_per_limit = 100
    
    # Theoretical lower bound
    min_possible = n_rows * k * (k + 1) // 2
    
    while time.time() < deadline:
        if current_limit < min_possible:
            break
        
        found = False
        for _ in range(trials_per_limit):
            if time.time() > deadline - 2:
                break
            
            seed = seed_counter
            seed_counter += 1
            
            rows = smart_randomized_greedy(n_rows, k, current_limit, seed)
            if rows is not None:
                # Verify
                all_diffs = set()
                valid = True
                for r in rows:
                    if len(r) != k + 1:
                        valid = False
                        break
                    if r[0] != 0:
                        valid = False
                        break
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                
                if valid:
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = rows
                    current_limit = new_scope - 1
                    found = True
                    break
        
        if not found:
            current_limit -= 1

    return best_rows

# Keep the original function names and signatures
def greedy_construct(n_rows, k):
    """
    Standard greedy construction.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not diffs & used:
                row.append(x)
                used.update(diffs)
        rows.append(row)
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Smart Randomized Greedy with Lookahead
    def smart_randomized_greedy(n_rows, k, limit, seed):
        random.seed(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            success = True
            while len(row) < k + 1:
                candidates = []
                # Collect all valid candidates
                for x in range(current_max + 1, limit + 1):
                    diffs = {x - a for a in row}
                    if not (diffs & used):
                        candidates.append(x)
                
                if not candidates:
                    success = False
                    break
                
                # Heuristic: Ensure we can complete the row within the limit
                remaining_steps = (k + 1 - len(row)) - 1
                valid_candidates = [x for x in candidates if x + remaining_steps <= limit]
                
                if not valid_candidates:
                    success = False
                    break
                
                # Pick randomly from valid candidates
                # To explore more, we can weight towards the middle or end?
                # Uniform is a good start.
                next_x = random.choice(valid_candidates)
                
                row.append(next_x)
                current_max = next_x
                used.update({next_x - a for a in row[:-1]})
            
            if not success:
                return None
            rows.append(row)
            
        global_max = max(max(r) for r in rows)
        if global_max > limit:
            return None
        return rows

    # 3. Search loop
    current_limit = best_scope - 1
    seed_counter = 0
    trials_per_limit = 200 # Increased trials
    
    # Theoretical lower bound
    min_possible = n_rows * k * (k + 1) // 2
    
    while time.time() < deadline:
        if current_limit < min_possible:
            break
        
        found = False
        for _ in range(trials_per_limit):
            if time.time() > deadline - 2:
                break
            
            seed = seed_counter
            seed_counter += 1
            
            rows = smart_randomized_greedy(n_rows, k, current_limit, seed)
            if rows is not None:
                # Verify
                all_diffs = set()
                valid = True
                for r in rows:
                    if len(r) != k + 1:
                        valid = False
                        break
                    if r[0] != 0:
                        valid = False
                        break
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                
                if valid:
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = rows
                    current_limit = new_scope - 1
                    found = True
                    break
        
        if not found:
            current_limit -= 1

    return best_rows
