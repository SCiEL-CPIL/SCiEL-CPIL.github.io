import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit, 10s safety margin
    
    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    TARGET_RANGE = list(range(TARGET_MIN, TARGET_MAX + 1))
    
    # State
    # A is a set of integers
    # mult[s] is the multiplicity of sum s
    # We only care about s in [128, 256] for the objective, 
    # but we need to track all sums? No, only those in target range affect the score directly.
    # However, coverage is only required for target range.
    # Sums outside target range don't matter for the objective, except they consume "capacity" 
    # if we were minimizing total pairs, but we are minimizing peak in target range.
    # So we can ignore sums outside [128, 256] for the objective calculation.
    
    def get_energy(mult, A):
        # mult is a dict or list for target range
        # We want to minimize peak, subject to coverage.
        # Penalty for uncovered: sum(max(0, 1 - c_s)^2)
        # Penalty for peak: max(c_s)
        # Let's use a smooth approximation for peak to help SA
        # Or just track max and use it.
        
        uncovered_penalty = 0
        max_mult = 0
        for s in TARGET_RANGE:
            c = mult.get(s, 0)
            if c < 1:
                uncovered_penalty += (1 - c) ** 2
            if c > max_mult:
                max_mult = c
        
        # Weighting: We want max_mult to be small.
        # If we find max_mult=2, score is 32.5.
        # If we find max_mult=1, score is 65.
        # The penalty should strongly discourage max_mult > 1.
        # Let's try: E = uncovered_penalty + 10 * max_mult
        # This might be too coarse.
        
        # Alternative: E = sum(c_s^2) for s in target range.
        # Minimizing sum of squares minimizes variance, which helps flatten the distribution.
        # If we minimize sum(c_s^2) subject to c_s >= 1, we get the flattest distribution.
        sum_sq = sum(mult.get(s, 0) ** 2 for s in TARGET_RANGE)
        
        # Combine:
        # If any c_s == 0, it's bad.
        # E = sum_sq + 100 * uncovered_count
        uncovered_count = sum(1 for s in TARGET_RANGE if mult.get(s, 0) == 0)
        
        return sum_sq + 100 * uncovered_count

    # Initialization
    # Try a few random initializations
    best_A = None
    best_score = float('inf')
    
    # We will run SA for a fixed number of iterations or until deadline
    # Let's do multiple restarts
    
    def run_sa(init_A):
        A = set(init_A)
        # Initialize mult
        mult = {}
        for a in A:
            for b in A:
                if a <= b:
                    s = a + b
                    if TARGET_MIN <= s <= TARGET_MAX:
                        mult[s] = mult.get(s, 0) + 1
        
        current_E = get_energy(mult, A)
        best_local_E = current_E
        best_local_A = set(A)
        
        T = 10.0
        cooling_rate = 0.99
        
        while T > 0.1:
            if time.time() > deadline - 5.0:
                break
            
            # Choose a move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_A = set(A)
            delta_E = 0
            
            if move_type == 'add':
                # Add a random element not in A
                candidates = [i for i in range(N+1) if i not in A]
                if not candidates: continue
                x = random.choice(candidates)
                new_A.add(x)
                
                # Update mult
                # Pairs involving x: {x, y} for y in A, and {x, x}
                # Note: A is the OLD set.
                # For each y in A:
                #   s = x + y. If s in target, mult[s] += 1
                #   s = 2x. If s in target, mult[s] += 1 (only once)
                
                # We need to calculate the change in E.
                # It's easier to just compute new E if changes are small, 
                # but get_energy is O(129). 
                # Updating mult is O(|A|). 
                # Let's just recompute E from scratch? 
                # 129 ops is fast.
                
                # But we need to update mult correctly for the next step.
                # Let's apply the change to mult, calculate new E, and if rejected, revert.
                
                old_mult = dict(mult)
                
                for y in A:
                    s = x + y
                    if TARGET_MIN <= s <= TARGET_MAX:
                        mult[s] = mult.get(s, 0) + 1
                s = 2*x
                if TARGET_MIN <= s <= TARGET_MAX:
                    mult[s] = mult.get(s, 0) + 1
                
                new_E = get_energy(mult, new_A)
                
                if new_E < current_E or random.random() < pow(2.718, -(new_E - current_E) / T):
                    A = new_A
                    current_E = new_E
                    if current_E < best_local_E:
                        best_local_E = current_E
                        best_local_A = set(A)
                else:
                    mult = old_mult
                    
            elif move_type == 'remove':
                if not A: continue
                x = random.choice(list(A))
                new_A.remove(x)
                
                old_mult = dict(mult)
                
                # Remove pairs involving x
                for y in A:
                    if y == x: continue # handled by 2x
                    s = x + y
                    if TARGET_MIN <= s <= TARGET_MAX:
                        mult[s] -= 1
                        if mult[s] == 0: del mult[s]
                        
                s = 2*x
                if TARGET_MIN <= s <= TARGET_MAX:
                    mult[s] -= 1
                    if mult[s] == 0: del mult[s]
                    
                new_E = get_energy(mult, new_A)
                
                if new_E < current_E or random.random() < pow(2.718, -(new_E - current_E) / T):
                    A = new_A
                    current_E = new_E
                    if current_E < best_local_E:
                        best_local_E = current_E
                        best_local_A = set(A)
                else:
                    mult = old_mult
                    
            elif move_type == 'swap':
                if not A: continue
                x = random.choice(list(A))
                candidates = [i for i in range(N+1) if i not in A]
                if not candidates: continue
                y = random.choice(candidates)
                
                new_A.remove(x)
                new_A.add(y)
                
                old_mult = dict(mult)
                
                # Remove x
                for z in A:
                    if z == x: continue
                    s = x + z
                    if TARGET_MIN <= s <= TARGET_MAX:
                        mult[s] -= 1
                        if mult[s] == 0: del mult[s]
                s = 2*x
                if TARGET_MIN <= s <= TARGET_MAX:
                    mult[s] -= 1
                    if mult[s] == 0: del mult[s]
                    
                # Add y
                # Note: A still contains x. We removed x from new_A, but mult update 
                # should be based on the set AFTER removal of x and BEFORE addition of y?
                # No, swap means we replace x with y.
                # The set of OTHER elements is A \ {x}.
                # Pairs removed: {x, z} for z in A\{x}, and {x, x}.
                # Pairs added: {y, z} for z in A\{x}, and {y, y}.
                
                other_elements = [z for z in A if z != x]
                
                for z in other_elements:
                    s = y + z
                    if TARGET_MIN <= s <= TARGET_MAX:
                        mult[s] = mult.get(s, 0) + 1
                        
                s = 2*y
                if TARGET_MIN <= s <= TARGET_MAX:
                    mult[s] = mult.get(s, 0) + 1
                    
                new_E = get_energy(mult, new_A)
                
                if new_E < current_E or random.random() < pow(2.718, -(new_E - current_E) / T):
                    A = new_A
                    current_E = new_E
                    if current_E < best_local_E:
                        best_local_E = current_E
                        best_local_A = set(A)
                else:
                    mult = old_mult
            
            T *= cooling_rate
        
        return best_local_A, best_local_E

    # Restart loop
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        # Random initialization
        # Size around 18-20
        K = random.randint(16, 22)
        init_A = random.sample(range(N+1), K)
        
        A, E = run_sa(init_A)
        
        # Check if this A is valid (covers range)
        # Calculate actual max multiplicity
        mult = {}
        for a in A:
            for b in A:
                if a <= b:
                    s = a + b
                    if TARGET_MIN <= s <= TARGET_MAX:
                        mult[s] = mult.get(s, 0) + 1
        
        is_valid = all(mult.get(s, 0) > 0 for s in TARGET_RANGE)
        if is_valid:
            max_m = max(mult.get(s, 0) for s in TARGET_RANGE)
            # Score is 65 / max_m
            # We want to minimize max_m
            if max_m < best_score:
                best_score = max_m
                best_A = A
                
        if best_score == 2: # Theoretical minimum
            break
        if best_score == 1:
            break

    if best_A is None:
        # Fallback to seed if no valid set found
        return list(range(64, 129))
        
    return list(best_A)
