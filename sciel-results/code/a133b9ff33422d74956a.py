import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    N = q ** n
    # Precompute balls
    balls = []
    for x in range(N):
        digits = [int(d) for d in str(x).zfill(n)] # Wait, base q representation
        # Better:
        digits = []
        tmp = x
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse() # MSB first? Doesn't matter as long as consistent.
        
        ball = [x]
        for i in range(n):
            for d in range(q):
                if d == digits[i]: continue
                new_digits = digits[:]
                new_digits[i] = d
                # Convert back to int
                val = 0
                for j in range(n):
                    val = val * q + new_digits[j]
                ball.append(val)
        balls.append(ball)
        
    # Greedy function
    def greedy_cover(initial_code=None, uncovered_set=None):
        # Returns a list of codewords
        if uncovered_set is None:
            uncovered_set = set(range(N))
        else:
            uncovered_set = set(uncovered_set)
            
        code = []
        if initial_code:
            code.extend(initial_code)
            # Update uncovered_set based on initial_code? 
            # It's easier to just start fresh if we pass initial_code as a starting point for *repair*
            # But for standard greedy, we start with empty.
            
        # For repair, we want to cover only `uncovered_set`.
        # But the codewords added must be distinct from existing? 
        # The problem says "codewords must be distinct".
        # If we are repairing, we might add a codeword already in the code? No, that's redundant.
        # So we should exclude existing code from candidates? Or just allow it, redundancy check will remove it.
        
        # Let's implement a generic "cover these words" that adds to a given code.
        # To keep it simple and fast, I'll implement a specific greedy that builds from scratch,
        # and a repair function.
        
        pass

    # Actually, let's just write a fast greedy from scratch.
    def run_greedy():
        covered = [False] * N
        coverage_count = [0] * N
        code = []
        
        while True:
            # Check if all covered
            # If any uncovered, find best codeword
            max_gain = -1
            best_c = -1
            
            # Optimization: Only check codewords that have potential to cover new words?
            # Hard to know without checking.
            # But N is small.
            
            for c in range(N):
                gain = 0
                for w in balls[c]:
                    if not covered[w]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
                    # Early break if gain == len(balls[c])? i.e. 15?
                    # If gain == 15, it's the max possible.
                    if gain == len(balls[c]):
                        break
            
            if max_gain == 0:
                # All covered
                break
                
            # Add best_c
            code.append(best_c)
            for w in balls[best_c]:
                if not covered[w]:
                    covered[w] = True
                coverage_count[w] += 1
                
        return code

    def remove_redundant(code):
        # Convert to set for fast lookup? 
        # We need to know coverage counts.
        # Rebuild coverage counts from scratch is O(|C| * B).
        # |C| ~ 200, B ~ 15. 3000 ops. Fast.
        
        changed = True
        while changed:
            changed = False
            new_code = []
            # Calculate coverage for current code
            cov_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cov_count[w] += 1
            
            for c in code:
                # Check if c is essential
                essential = False
                for w in balls[c]:
                    if cov_count[w] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove c
                    changed = True
                    # We can't just skip it in new_code, we need to update cov_count for next iterations?
                    # It's easier to just build new_code and restart the while loop.
                    # Since changed=True, the while loop will re-calculate cov_count.
                    continue
                else:
                    new_code.append(c)
                    
            code = new_code
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple times
    while time.time() < deadline:
        # 1. Standard Greedy
        code = run_greedy()
        code = remove_redundant(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # 2. Perturbed Greedy
        # Start with a random subset of the best code? Or random start?
        # Let's try: Take best_code, remove k random, repair.
        if best_code is not None and len(best_code) > 0:
            k = random.randint(1, 3)
            # Pick k indices to remove
            indices_to_remove = random.sample(range(len(best_code)), min(k, len(best_code)))
            # Create a set of removed codewords
            removed_set = set([best_code[i] for i in indices_to_remove])
            
            # Determine uncovered words
            # We need to know which words were ONLY covered by these removed codewords.
            # This requires knowing the coverage of the original best_code.
            # Let's just re-calculate coverage of best_code, subtract removed ones.
            
            # Simpler: 
            # Start with code = best_code - removed_set.
            # Calculate uncovered words.
            # Run greedy to cover them.
            
            current_code = [c for i, c in enumerate(best_code) if i not in indices_to_remove]
            
            # Calculate covered
            covered = [False] * N
            for c in current_code:
                for w in balls[c]:
                    covered[w] = True
            
            # Find uncovered
            uncovered = [w for w in range(N) if not covered[w]]
            
            if not uncovered:
                # Still covered? Then we just removed redundant ones (which remove_redundant should have caught, but maybe interaction?)
                # Actually, if we remove non-redundant ones, they will be uncovered.
                pass
                
            # Greedy repair
            # We want to add codewords to cover `uncovered`.
            # We can reuse the logic from run_greedy but restricted to covering `uncovered`.
            # However, adding a codeword might cover words already covered, that's fine.
            # We just need to minimize the number of *new* codewords.
            
            # Let's implement a specific repair greedy
            while uncovered:
                max_gain = -1
                best_c = -1
                for c in range(N):
                    if c in current_code: # Avoid adding duplicates? 
                         # Actually, adding a duplicate is useless.
                         # But checking `c in current_code` is O(|C|). 
                         # Using a set for current_code is better.
                         pass
                    gain = 0
                    for w in balls[c]:
                        if not covered[w]:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_c = c
                        if gain == len(balls[c]): break
                
                if max_gain == 0:
                    break # Should not happen if uncovered is not empty
                
                current_code.append(best_c)
                for w in balls[best_c]:
                    covered[w] = True
                    
                # Update uncovered list? 
                # It's easier to just re-scan or maintain a set.
                # Since N is small, re-scanning `covered` to find `uncovered` is $O(N)$.
                # Steps in repair are small.
                uncovered = [w for w in range(N) if not covered[w]]
                
            # After repair, run remove_redundant
            current_code = remove_redundant(current_code)
            
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]
                
    # Convert best_code to list of lists of ints
    result = []
    for c in best_code:
        digits = []
        tmp = c
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        result.append(digits)
        
    return result
