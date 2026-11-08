import time
import random
from math import log, floor

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
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
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    best_score = score_set(best_set)
    
    # Strategy: Digit-based parameterization
    # U is defined by digits d_0=0, d_1, ..., d_{k-1} in base B.
    # U = { sum_{i=1}^{k-1} d_i * B^i }
    # We search over B and the digit sequences.
    
    def generate_from_digits(digits, base):
        """Generate the set U from a list of digits and a base."""
        U = [0]
        current = 0
        for i in range(1, len(digits)):
            current += digits[i] * (base ** i)
            if current not in U:
                U.append(current)
        return sorted(U)

    # Candidate bases
    bases = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 30]
    
    best_overall_set = best_set
    best_overall_score = best_score
    
    # We will run a local search on the digit space for each base.
    # The "state" is a list of digits.
    
    for base in bases:
        if time.time() > deadline - 10:
            break
            
        # Determine max depth based on base
        # We want to keep max(U) manageable but large enough to explore structure.
        # max U approx base^k.
        # If base is small, we can go deeper.
        max_depth = 15 if base <= 3 else 12 if base <= 5 else 10 if base <= 10 else 7
        
        # Initial digit sets:
        initial_digit_sets = []
        
        # 1. All 1s
        d_ones = [0] * max_depth
        for i in range(1, max_depth):
            d_ones[i] = 1
        initial_digit_sets.append(d_ones)
        
        # 2. All (base-1)s
        d_max = [0] * max_depth
        for i in range(1, max_depth):
            d_max[i] = base - 1
        initial_digit_sets.append(d_max)
        
        # 3. Alternating
        d_alt = [0] * max_depth
        for i in range(1, max_depth):
            d_alt[i] = 1 if i % 2 == 1 else 0
        initial_digit_sets.append(d_alt)
        
        # 4. Random
        for _ in range(5):
            d_rand = [0] * max_depth
            for i in range(1, max_depth):
                d_rand[i] = random.randint(0, base - 1)
            initial_digit_sets.append(d_rand)
            
        # 5. Sparse random (mostly 0s)
        for _ in range(5):
            d_sparse = [0] * max_depth
            for i in range(1, max_depth):
                if random.random() < 0.3:
                    d_sparse[i] = random.randint(1, base - 1)
            initial_digit_sets.append(d_sparse)

        for init_digits in initial_digit_sets:
            if time.time() > deadline - 5:
                break
                
            current_digits = list(init_digits)
            current_U = generate_from_digits(current_digits, base)
            current_score = score_set(current_U)
            
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_set = current_U
            
            # Local search on digits
            no_improve = 0
            max_no_improve = 30
            
            while no_improve < max_no_improve and time.time() < deadline - 5:
                improved = False
                
                # Try changing one digit
                pos = random.randint(1, max_depth - 1)
                old_val = current_digits[pos]
                
                # Try changing to a different digit
                if base <= 10:
                    new_vals = [v for v in range(base) if v != old_val]
                    random.shuffle(new_vals)
                    # Try top few
                    for new_val in new_vals[:5]:
                        new_digits = list(current_digits)
                        new_digits[pos] = new_val
                        new_U = generate_from_digits(new_digits, base)
                        new_score = score_set(new_U)
                        
                        if new_score > current_score:
                            current_digits = new_digits
                            current_U = new_U
                            current_score = new_score
                            improved = True
                            no_improve = 0
                            
                            if current_score > best_overall_score:
                                best_overall_score = current_score
                                best_overall_set = current_U
                            break
                else:
                    # For larger bases, just try a few random perturbations
                    for _ in range(5):
                        new_val = random.randint(0, base - 1)
                        if new_val == old_val:
                            continue
                        new_digits = list(current_digits)
                        new_digits[pos] = new_val
                        new_U = generate_from_digits(new_digits, base)
                        new_score = score_set(new_U)
                        
                        if new_score > current_score:
                            current_digits = new_digits
                            current_U = new_U
                            current_score = new_score
                            improved = True
                            no_improve = 0
                            
                            if current_score > best_overall_score:
                                best_overall_score = current_score
                                best_overall_set = current_U
                            break
                
                if not improved:
                    no_improve += 1
                
                # Occasionally try changing two digits
                if random.random() < 0.3 and base <= 8:
                    pos1 = random.randint(1, max_depth - 2)
                    pos2 = pos1 + 1
                    v1 = random.randint(0, base - 1)
                    v2 = random.randint(0, base - 1)
                    
                    new_digits = list(current_digits)
                    new_digits[pos1] = v1
                    new_digits[pos2] = v2
                    new_U = generate_from_digits(new_digits, base)
                    new_score = score_set(new_U)
                    
                    if new_score > current_score:
                        current_digits = new_digits
                        current_U = new_U
                        current_score = new_score
                        improved = True
                        no_improve = 0
                        
                        if current_score > best_overall_score:
                            best_overall_score = current_score
                            best_overall_set = current_U

    # Final refinement: Try to add or remove elements from the best set found
    # This allows breaking the strict digit structure if it helps.
    if time.time() < deadline - 2:
        current = list(best_overall_set)
        current_score = best_overall_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 100:
            improved = False
            iteration += 1
            n = len(current)
            
            # Try removing an element
            if n > 3:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
            
            # Try adding an element
            if not improved and n < 20:
                # Try adding small integers or multiples of existing
                candidates = set()
                max_val = max(current)
                # Small range around existing elements
                for x in current:
                    for d in range(-5, 6):
                        if x + d > 0:
                            candidates.add(x + d)
                # Also try some random large ones
                for _ in range(10):
                    candidates.add(random.randint(1, max_val * 2))
                
                random.shuffle(list(candidates))
                for c in list(candidates)[:20]:
                    if c in current:
                        continue
                    new_set = sorted(current + [c])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
            
            # Try replacing an element with a nearby value
            if not improved and n > 1:
                for i in range(1, n):
                    old_val = current[i]
                    for d in range(-3, 4):
                        if d == 0:
                            continue
                        new_val = old_val + d
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

        if current_score > best_overall_score:
            best_overall_set = current

    return best_overall_set

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
