import time
import random
from math import log

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
    deadline = start_time + 110

    best_score = -1
    best_set = [0, 1]
    
    def get_score(U):
        return score_set(U)

    def generate_from_digits(digits, base):
        """
        digits: list of integers, digits[0] is ignored or 0.
        U = { sum_{i=1}^{k-1} d_i * base^i }
        """
        vals = [0]
        for i in range(1, len(digits)):
            if digits[i] != 0:
                vals.append(sum(digits[j] * (base ** j) for j in range(1, i + 1)))
        # Ensure unique and sorted
        return sorted(set(vals))

    # Strategy: Search over bases and digit lengths
    # Score is 1 + log(ratio)/log(2*max+1)
    # To maximize, we want high ratio and small max(U).
    # Small max(U) implies small base or small length.
    # High ratio implies specific digit structures.
    
    # Try small bases and lengths
    # Base 2: digits are 0/1.
    # Base 3: digits 0/1/2.
    
    best_found = -1
    best_config = None

    # We will perform a local search in the space of (base, digits)
    # Since base is discrete, we loop over bases.
    
    for base in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
            
        # Max length k such that max(U) is not too large
        # max(U) ~ base^k
        # If base=2, k=15 -> 32768. log(65537) ~ 16.
        # If base=10, k=5 -> 100000. log(200001) ~ 18.
        # The log in denominator grows slowly.
        # The ratio is bounded.
        
        # Let's try lengths up to 12 for base 2, 8 for base 3, etc.
        max_len = 15 if base == 2 else 10 if base == 3 else 8 if base == 4 else 6
        
        # Initialize with a few random configurations
        configs = []
        
        # 1. All 1s (except pos 0)
        d_ones = [0] * max_len
        for i in range(1, max_len):
            d_ones[i] = 1
        configs.append(d_ones)
        
        # 2. Alternating 1, 0
        d_alt = [0] * max_len
        for i in range(1, max_len):
            d_alt[i] = 1 if i % 2 == 1 else 0
        configs.append(d_alt)
        
        # 3. Random
        for _ in range(5):
            d_rand = [0] * max_len
            for i in range(1, max_len):
                d_rand[i] = random.randint(0, base - 1)
            configs.append(d_rand)

        for init_d in configs:
            if time.time() > deadline - 5:
                break
                
            current_d = list(init_d)
            current_U = generate_from_digits(current_d, base)
            current_score = get_score(current_U)
            
            if current_score > best_found:
                best_found = current_score
                best_config = (base, current_d)

            # Local Search: Steepest Descent
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try changing each position to each possible digit
                # To speed up, we can randomize the order of positions
                positions = list(range(1, max_len))
                random.shuffle(positions)
                
                for pos in positions:
                    if time.time() > deadline - 5:
                        break
                    best_val = current_d[pos]
                    best_s = current_score
                    
                    # Try all other digits
                    for val in range(base):
                        if val == best_val:
                            continue
                        new_d = current_d[:]
                        new_d[pos] = val
                        new_U = generate_from_digits(new_d, base)
                        s = get_score(new_U)
                        if s > best_s:
                            best_s = s
                            best_val = val
                    
                    if best_val != current_d[pos]:
                        current_d[pos] = best_val
                        current_score = best_s
                        current_U = generate_from_digits(current_d, base)
                        improved = True
                        
                        if current_score > best_found:
                            best_found = current_score
                            best_config = (base, current_d)
                        
                        # If we improved, restart the position loop (or just continue, 
                        # but restarting ensures we don't miss interactions)
                        # For efficiency, just continue to next position in this pass
                        # and do another pass if improved.
                        break # Break inner loop, restart outer while loop? 
                        # Actually, standard steepest descent picks the BEST move in one pass.
                        # Let's stick to: find best single move, apply it, repeat.
                        # So the break above is correct for "apply best move and restart".
                        # But we need to find the best move among ALL positions first.
                        # My current loop breaks on the FIRST position that has an improvement.
                        # This is "First-Improvement" or "Hill Climbing".
                        # Steepest Descent is better. Let's switch to Steepest Descent.

            # Redo local search as Steepest Descent for robustness
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                best_move = None
                best_move_score = current_score
                
                for pos in range(1, max_len):
                    for val in range(base):
                        if val == current_d[pos]:
                            continue
                        new_d = current_d[:]
                        new_d[pos] = val
                        new_U = generate_from_digits(new_d, base)
                        s = get_score(new_U)
                        if s > best_move_score:
                            best_move_score = s
                            best_move = (pos, val)
                
                if best_move:
                    pos, val = best_move
                    current_d[pos] = val
                    current_score = best_move_score
                    current_U = generate_from_digits(current_d, base)
                    improved = True
                    
                    if current_score > best_found:
                        best_found = current_score
                        best_config = (base, current_d)
                    else:
                        # This shouldn't happen if best_move_score > current_score
                        pass

    # If we found a good config, return the set
    if best_config:
        b, d = best_config
        best_set = generate_from_digits(d, b)
    else:
        # Fallback
        best_set = [0, 1, 4, 9]
        
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
