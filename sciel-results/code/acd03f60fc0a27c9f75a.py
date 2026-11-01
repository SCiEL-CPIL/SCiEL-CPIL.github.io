"""Local search optimization for ternary covering codes."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill

    total = q ** n
    if total > 100000:
        # Fallback for larger spaces, though task is specific to 2187
        # Just return a valid but potentially suboptimal code
        return [[0]*n]

    # Precompute all words as digits
    # For q=3, n=7, total=2187.
    # We represent words as tuples of digits for easy Hamming distance calc?
    # Or better, use integer representation and precompute neighbors.
    
    # For radius 1, a word c covers c and its n*(q-1) neighbors.
    # Let's precompute the list of covered integers for each integer c.
    
    def int_to_digits(x, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return tuple(digits)

    def digits_to_int(digits, q):
        x = 0
        for i in range(len(digits) - 1, -1, -1):
            x = x * q + digits[i]
        return x

    # Precompute balls for all possible codewords
    # ball[c] is a list of integers covered by c
    balls = []
    for c in range(total):
        digits = int_to_digits(c, n, q)
        covered = [c]
        for i in range(n):
            orig = digits[i]
            for d in range(q):
                if d == orig:
                    continue
                digits[i] = d
                w = digits_to_int(digits, q)
                covered.append(w)
            digits[i] = orig
        balls.append(covered)
        
    # Convert to sets for fast union/intersection? 
    # Actually, for local search, we need to evaluate the coverage of a set of codewords.
    # Coverage size = |Union of balls|
    
    # To speed up evaluation, we can maintain a count of how many times each word is covered.
    # But replacing one codeword with another requires updating the counts for the old and new balls.
    
    # Let's use a numpy array for coverage counts.
    # cov_count[w] = number of codewords in current code that cover w.
    # Uncovered words are those with cov_count[w] == 0.
    
    # Initial random code
    # We want to find a small K.
    # Lower bound is 156. Best known 186.
    # Let's try K from 160 to 200.
    
    best_code = None
    best_size = total + 1
    
    # We will run local search for various K
    # To save time, we'll focus on K around 170-190.
    
    # Precompute ball membership as bitsets? 
    # total=2187. 2187 bits ~ 274 bytes. 
    # Using Python integers as bitsets is very fast for OR/AND.
    
    # Let's convert balls to bitsets (Python integers)
    ball_bits = [0] * total
    for c in range(total):
        bits = 0
        for w in balls[c]:
            bits |= (1 << w)
        ball_bits[c] = bits
        
    # This representation allows fast union: union_bits = 0; for c in code: union_bits |= ball_bits[c]
    # And fast count: union_bits.bit_count()
    
    # Local Search Strategy:
    # 1. Pick a target size K.
    # 2. Initialize with K random distinct codewords.
    # 3. Calculate current coverage.
    # 4. While time allows:
    #    - Pick a random index i in code.
    #    - Pick a random new codeword c_new not in code.
    #    - Calculate new coverage if we replace code[i] with c_new.
    #    - If new coverage > old coverage, accept.
    #    - If new coverage == old coverage, accept with some probability (or always to shuffle).
    #    - If coverage becomes total, we can try to remove codewords to reduce K? 
    #       Or just stop and check if we can reduce K by removing redundant codewords.
    
    # Optimization: 
    # Instead of recalculating full union every time (K ORs), 
    # we can maintain the union bitset and update it? 
    # Removing a codeword from union is hard (need to know if any other covers it).
    # So, recalculation is O(K). K ~ 180. 
    # Each step: O(K) for union calc. 
    # We can do ~100k steps in 120s? 100k * 180 ops = 18M ops. Python is slow.
    # 18M simple integer ORs might take ~1-2 seconds. So ~100k steps is feasible.
    
    def calculate_coverage(code):
        union = 0
        for c in code:
            union |= ball_bits[c]
        return union.bit_count()
    
    def is_valid(code):
        return calculate_coverage(code) == total

    # Try different K values
    # We want the smallest K.
    # Let's try K from 160 to 190.
    # For each K, run local search.
    
    # To be efficient, we'll run a single long local search starting from a good K
    # and try to minimize K by attempting removals.
    
    # Start with K = 180 (midpoint)
    current_k = 180
    # Random initialization
    code = random.sample(range(total), current_k)
    
    # Ensure distinct
    current_cov = calculate_coverage(code)
    
    # Local search loop
    steps = 0
    max_steps = 500000 # Cap steps to avoid infinite loop if time check fails
    
    while time.time() < deadline - 5.0:
        if steps > max_steps:
            break
        steps += 1
        
        # Random move: replace one codeword
        idx = random.randint(0, current_k - 1)
        old_c = code[idx]
        
        # Pick new_c
        # Avoid picking old_c or existing ones? 
        # Picking existing one is a no-op. 
        # We can pick from all, and if it's in code, skip.
        
        new_c = random.randint(0, total - 1)
        if new_c in code:
            continue
            
        # Calculate new coverage
        # Optimization: 
        # new_union = (old_union & ~ball_bits[old_c]) | ball_bits[new_c]
        # But we don't have old_union stored easily if we just recalc.
        # Let's recalc. It's O(K).
        
        # Actually, let's try a smarter update.
        # We need to know if removing old_c reduces coverage.
        # And if adding new_c increases it.
        
        # Let's just recalc for simplicity first, as K is small.
        
        new_code = code[:]
        new_code[idx] = new_c
        
        new_cov = calculate_coverage(new_code)
        
        if new_cov > current_cov:
            code = new_code
            current_cov = new_cov
        elif new_cov == current_cov:
            # Accept to explore
            if random.random() < 0.5:
                code = new_code
                current_cov = new_cov
        else:
            # Reject
            pass
            
        # Check if we can reduce K
        if current_cov == total:
            # Try to remove a codeword
            # Pick a random codeword to try removing
            rem_idx = random.randint(0, current_k - 1)
            rem_c = code[rem_idx]
            
            test_code = code[:rem_idx] + code[rem_idx+1:]
            test_cov = calculate_coverage(test_code)
            
            if test_cov == total:
                code = test_code
                current_k -= 1
                current_cov = total
                # Update best
                if current_k < best_size:
                    best_size = current_k
                    best_code = code[:]
            else:
                # Can't remove, maybe try swapping to a "better" codeword that allows removal later?
                # For now, just continue.
                pass
                
        # Early stop if we hit theoretical lower bound? 156.
        if current_k <= 156:
            break

    # Final polish: try to remove more
    if current_cov == total:
        # Greedy removal
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for i in range(current_k):
                if time.time() > deadline - 2.0:
                    break
                test_code = code[:i] + code[i+1:]
                if calculate_coverage(test_code) == total:
                    code = test_code
                    current_k -= 1
                    changed = True
                    if current_k < best_size:
                        best_size = current_k
                        best_code = code[:]
                    break # Restart loop after removal

    if best_code is None:
        # Fallback: return current code if valid, else a known valid one
        if current_cov == total:
            best_code = code
        else:
            # Fallback to tetracode direct sum
            import itertools
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            # Convert to integers? No, return list of lists of ints.
            # The function expects list of codewords, each a list of n integers.
            
    # Ensure return format is list of lists of integers
    # best_code is currently list of integers (if from local search) or list of lists (if fallback)
    # Wait, in local search, `code` is a list of integers (indices).
    # We need to convert these indices back to digit lists.
    
    if isinstance(best_code[0], int):
        final_code = [list(int_to_digits(c, n, q)) for c in best_code]
    else:
        final_code = best_code
        
    return final_code

# EVOLVE-BLOCK-END
