import time
import random
import math

def score_set(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    if not U:
        return -1
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    # Optimization: Use lists for speed, then convert to sets
    # But for small n, sets are fine.
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            if d != 0:
                diffs.add(d)
                diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + math.log(ratio) / math.log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11] # Fallback
    
    # Strategy: Base-b digit optimization
    # We represent U as { sum d_k * b^k | d_k in D_k }
    # where D_k is a set of allowed digits for position k.
    # We optimize the digit sets D_k and the base b.
    
    # 1. Generate initial diverse candidates using this structure
    candidates = []
    
    # Fixed base candidates
    for b in [2, 3, 4, 5, 6]:
        # Try different digit set sizes
        for max_digits in [1, 2, 3]:
            # Start with simple digit sets {0, 1} or {0, 1, 2} etc.
            # We will refine these.
            pass # Handled in optimization loop below
            
    # 2. Optimization Loop
    # We will try several random configurations of (base, number_of_positions, initial_digit_sets)
    # and refine them using local search on the digit choices.
    
    num_trials = 200
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        # Randomize base and structure
        b = random.randint(2, 8)
        num_positions = random.randint(3, 8)
        
        # Initial digit sets: usually {0, 1} or {0, 1, 2}
        # Let's start with {0, 1} for all positions, which gives powers of b.
        # Then we might expand digit sets.
        
        # Represent state as list of sets of digits
        digit_sets = [set([0, 1]) for _ in range(num_positions)]
        
        # Function to expand digit sets to a concrete set U
        def expand(digits, base):
            U = set([0])
            for k, d_set in enumerate(digits):
                pow_b = base ** k
                # Cartesian product logic
                new_U = set()
                for u in U:
                    for d in d_set:
                        new_U.add(u + d * pow_b)
                U = new_U
            return sorted(list(U))
        
        current_U = expand(digit_sets, b)
        current_score = score_set(current_U)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_U)
            
        # Local Search on Digit Sets
        # Operations:
        # 1. Add a digit to a position
        # 2. Remove a digit from a position (if > 1 digit)
        # 3. Change base (small step)
        
        T = 0.1
        cooling = 0.95
        steps_per_trial = 50
        
        for step in range(steps_per_trial):
            if time.time() > deadline - 5:
                break
                
            op = random.random()
            
            # 1. Modify digit sets
            if op < 0.7:
                pos = random.randint(0, num_positions - 1)
                if random.random() < 0.5:
                    # Add digit
                    # Try adding a small digit
                    max_d = max(digit_sets[pos]) if digit_sets[pos] else 0
                    # Limit digit size to keep U manageable
                    if max_d < b: 
                        new_d = random.randint(max_d + 1, b)
                        if new_d not in digit_sets[pos]:
                            digit_sets[pos].add(new_d)
                else:
                    # Remove digit
                    if len(digit_sets[pos]) > 1:
                        # Don't remove 0 if it's the only way to keep 0 in U? 
                        # Actually 0 is always in U because 0 is in all digit sets initially.
                        # If we remove 0 from a digit set, we might lose 0?
                        # No, 0 = 0*b^0 + 0*b^1... so 0 must be in all digit sets to keep 0 in U.
                        # Wait, if 0 is not in D_k, can we still form 0?
                        # Yes, if all other digits are 0. But if D_k doesn't contain 0, we can't pick 0 for that position.
                        # So 0 MUST be in every D_k.
                        if 0 in digit_sets[pos]:
                            # Cannot remove 0
                            pass
                        else:
                            # Remove random non-zero
                            d_to_remove = random.choice([d for d in digit_sets[pos] if d != 0])
                            digit_sets[pos].remove(d_to_remove)
            
            # 2. Change Base
            elif op < 0.9:
                # Small change to base
                delta_b = random.choice([-1, 1])
                new_b = b + delta_b
                if new_b >= 2:
                    b = new_b
            
            # 3. Add/Remove Position
            else:
                if random.random() < 0.5 and num_positions < 10:
                    # Add position
                    digit_sets.append(set([0, 1]))
                    num_positions += 1
                elif num_positions > 3:
                    # Remove position (keep the lowest ones? or random?)
                    # Removing a high position reduces max(U) significantly, which helps the denominator log(2*max(U)+1).
                    # Let's remove the highest position usually.
                    digit_sets.pop()
                    num_positions -= 1
            
            new_U = expand(digit_sets, b)
            new_score = score_set(new_U)
            
            # Acceptance
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_U)
                current_U = new_U
            else:
                # Metropolis
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_U = new_U
                        current_score = new_score
            
            T *= cooling
            
        # End local search for this trial

    # Final Polish: Standard local search on the best set found
    # This fixes any structural artifacts from the digit representation
    # e.g. if a digit was removed, the set might have "holes" that can be filled by shifting elements.
    
    if time.time() < deadline - 2:
        current_set = list(best_set)
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            n = len(current_set)
            
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try small deltas
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved and n < 30 and time.time() < deadline - 1:
                max_val = max(current_set)
                # Try values near the gap or max
                # Focus on gaps in the difference set or sum set?
                # Just random sampling in range for now
                for _ in range(50):
                    new_val = random.randint(1, max_val + 10)
                    if new_val in current_set:
                        continue
                    new_set = sorted(current_set + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
                        
            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current_set[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current_set

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
