import time
import random
import math

def construct():
    fixed = {1, 2, 5, 11}
    N = 284
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Helper to compute sums and conflicts efficiently
    def compute_sums_and_conflicts(S):
        """
        Returns (sums_set, conflict_count, size)
        sums_set: set of all pairwise sums a+b (a<=b)
        conflict_count: number of duplicate sums (i.e., total_sums_generated - len(sums_set))
        """
        s_list = sorted(S)
        n = len(s_list)
        sums = set()
        total_sums = 0
        for i in range(n):
            for j in range(i, n):
                s = s_list[i] + s_list[j]
                total_sums += 1
                sums.add(s)
        conflicts = total_sums - len(sums)
        return sums, conflicts, n

    def is_valid(S):
        # For the final answer, we just need to ensure it's valid
        _, conflicts, _ = compute_sums_and_conflicts(S)
        return conflicts == 0

    # Initial Greedy Construction
    def greedy_init(strategy):
        S = set(fixed)
        sums, _, _ = compute_sums_and_conflicts(S)
        
        if strategy == 'small':
            candidates = list(range(1, N + 1))
        elif strategy == 'large':
            candidates = list(range(N, 0, -1))
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
        for x in candidates:
            if x in S:
                continue
            # Check if adding x creates conflicts
            # New sums: {x + y for y in S} U {2x}
            new_sums = {x + y for y in S} | {2 * x}
            if not (new_sums & sums):
                S.add(x)
                sums |= new_sums
                
        return S

    # Generate an initial valid set
    best_set = set(fixed)
    best_size = len(fixed)
    
    # Try a few greedy inits to get a good starting point
    for strat in ['small', 'large', 'random', 'random']:
        S = greedy_init(strat)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Simulated Annealing
    current_set = best_set.copy()
    current_sums, current_conflicts, current_size = compute_sums_and_conflicts(current_set)
    
    # Score function: We want to maximize size, minimize conflicts.
    # Score = size * 1000 - conflicts
    # This ensures that a valid set of size 18 (18000) is better than an invalid set of size 19 (19000 - 100 = 18900)
    # Wait, if conflicts are small, 19000 - 100 = 18900 > 18000. So it might prefer invalid 19.
    # This is good for escaping local optima.
    
    def get_score(S, sums, conflicts, size):
        return size * 1000 - conflicts

    current_score = get_score(current_set, current_sums, current_conflicts, current_size)
    
    best_score = current_score
    best_solution = current_set.copy()
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    
    # Pre-compute all possible pairs? No, N=284 is small enough to check on the fly for moves.
    # But we can pre-compute the list of all numbers 1..284.
    all_numbers = list(range(1, N + 1))
    
    steps = 0
    max_steps = 50000  # Hard cap on steps to ensure we finish
    stall_counter = 0
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Choose a move type
        # 1. Add (if size < 20)
        # 2. Remove (if size > 4)
        # 3. Swap
        move_type = random.random()
        
        old_set = current_set.copy()
        old_sums = current_sums
        old_conflicts = current_conflicts
        old_size = current_size
        old_score = current_score
        
        if move_type < 0.3:
            # Add a random element
            if current_size >= 20:
                continue
            x = random.choice(all_numbers)
            if x in current_set:
                continue
            # Tentative add
            new_set = current_set | {x}
            new_sums, new_conflicts, new_size = compute_sums_and_conflicts(new_set)
            new_score = get_score(new_set, new_sums, new_conflicts, new_size)
            
        elif move_type < 0.5:
            # Remove a random non-fixed element
            removable = [y for y in current_set if y not in fixed]
            if not removable:
                continue
            y = random.choice(removable)
            new_set = current_set - {y}
            new_sums, new_conflicts, new_size = compute_sums_and_conflicts(new_set)
            new_score = get_score(new_set, new_sums, new_conflicts, new_size)
            
        else:
            # Swap: Remove a non-fixed, Add a random
            removable = [y for y in current_set if y not in fixed]
            if not removable:
                continue
            y = random.choice(removable)
            x = random.choice(all_numbers)
            if x == y or x in current_set:
                continue
            new_set = (current_set - {y}) | {x}
            new_sums, new_conflicts, new_size = compute_sums_and_conflicts(new_set)
            new_score = get_score(new_set, new_sums, new_conflicts, new_size)

        delta = new_score - old_score
        
        # Acceptance criteria
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept the move
            current_set = new_set
            current_sums = new_sums
            current_conflicts = new_conflicts
            current_size = new_size
            current_score = new_score
            
            # Check if this is a valid solution with a better size
            if current_conflicts == 0 and current_size > best_size:
                best_size = current_size
                best_solution = current_set.copy()
                best_score = current_score
                stall_counter = 0
            else:
                if current_score > best_score:
                    best_score = current_score
                    # Don't update best_solution if it's invalid, but track score
                    # Actually, we only care about valid solutions for the final answer.
                    # But keeping track of best_score helps with annealing logic?
                    # No, the objective is to find a VALID max size set.
                    # So we only update best_solution if it is valid.
                stall_counter += 1
        else:
            stall_counter += 1
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # If stuck for too long, perform a larger perturbation (kick)
        if stall_counter > 500:
            # Remove 2-3 random non-fixed elements
            k = random.randint(2, 3)
            removable = [y for y in current_set if y not in fixed]
            if len(removable) >= k:
                to_remove = random.sample(removable, k)
                for y in to_remove:
                    current_set.remove(y)
                current_sums, current_conflicts, current_size = compute_sums_and_conflicts(current_set)
                current_score = get_score(current_set, current_sums, current_conflicts, current_size)
            stall_counter = 0
            # Reheat slightly
            T = max(T, 20.0)

    # Ensure the best solution found is valid
    # It should be, but just in case
    if not is_valid(best_solution):
        # Fallback to the greedy solution if SA failed to find a valid one (unlikely)
        best_solution = best_set
        
    return list(best_solution)
