"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_score = -1
    best_set = [0, 1, 3]  # Fallback
    
    def compute_score_golomb(ruler):
        """
        Compute score for a set U, assuming it is a Golomb ruler (distinct differences).
        If it's not a Golomb ruler, it returns a very low score to discourage non-Golomb sets,
        or we can just compute the exact score. 
        For speed in the search loop, we assume Golomb property holds or check it cheaply.
        """
        U = sorted(ruler)
        if not U or U[0] != 0:
            return -1
        n = len(U)
        if n < 2:
            return 0
        
        max_u = U[-1]
        if max_u == 0:
            return 0
        
        # Check if it's a Golomb ruler (optional, for safety)
        # In the search, we maintain the property, so this is just a final check
        diffs = set()
        for i in range(n):
            for j in range(i):
                d = U[i] - U[j]
                if d in diffs:
                    return -1 # Not a Golomb ruler, penalize
                diffs.add(d)
        
        # |U-U| for Golomb ruler is n^2 - n + 1
        n_diff = n * n - n + 1
        
        # |U+U| for Golomb ruler: 
        # Sums are distinct? No, sums can collide.
        # However, for sparse sets, |U+U| is roughly n^2/2.
        # We must compute |U+U| exactly.
        sumset = set()
        for i in range(n):
            for j in range(n):
                sumset.add(U[i] + U[j])
        n_sum = len(sumset)
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def is_golomb(ruler):
        """Check if ruler is a valid Golomb ruler."""
        n = len(ruler)
        diffs = set()
        for i in range(n):
            for j in range(i):
                d = ruler[i] - ruler[j]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(ruler):
        """Get set of pairwise differences."""
        n = len(ruler)
        diffs = set()
        for i in range(n):
            for j in range(i):
                diffs.add(ruler[i] - ruler[j])
        return diffs

    # Phase 1: Generate initial Golomb rulers using greedy or known constructions
    # We will try to find minimal span Golomb rulers for n = 3 to 12
    
    def greedy_golomb(n, max_span=1000):
        """Greedy construction of a Golomb ruler."""
        if n <= 1:
            return [0]
        if n == 2:
            return [0, 1]
        
        ruler = [0, 1]
        diffs = {1}
        
        for _ in range(n - 2):
            found = False
            for x in range(2, max_span + 1):
                # Check if adding x creates a collision
                valid = True
                new_diffs = set()
                for r in ruler:
                    d = x - r
                    if d in diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    ruler.append(x)
                    diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                break
        
        if len(ruler) == n:
            return ruler
        return None

    def optimize_golomb(ruler, max_iters=5000):
        """
        Local search to minimize the span (max element) of a Golomb ruler.
        This maximizes the score because the denominator log(2*max+1) decreases.
        """
        current = list(ruler)
        n = len(current)
        current_score = compute_score_golomb(current)
        best_score_local = current_score
        best_ruler = list(current)
        
        # Precompute diffs for fast update? 
        # For n <= 12, n^2 is small, recomputing is fast.
        
        for i in range(max_iters):
            # Try moving a pin
            # Pick a random index (not 0)
            idx = random.randint(1, n - 1)
            old_val = current[idx]
            
            # Try small shifts
            found_move = False
            # Try shifting left
            for delta in range(1, 10):
                new_val = old_val - delta
                if new_val < 0:
                    continue
                if new_val in current:
                    continue
                
                # Check validity
                valid = True
                # Remove old diffs involving idx? 
                # Simpler: just check the new pin against existing pins
                for j in range(n):
                    if j == idx:
                        continue
                    d = abs(current[j] - new_val)
                    # Check if this difference already exists between other pairs
                    # This is O(n^2) check.
                    # We need to check if d is in diffs of U\{current[idx]} U {new_val}
                    # Actually, just check if d is in diffs of U\{current[idx]}
                    # AND if new_val creates new collisions with other pins.
                    
                    # Let's do a full check for simplicity and correctness for n<15
                    # It's fast enough.
                    pass
                
                # Full check
                test_ruler = list(current)
                test_ruler[idx] = new_val
                test_ruler.sort()
                
                # Check distinctness
                diffs_set = set()
                ok = True
                for a in range(n):
                    for b in range(a):
                        d = test_ruler[a] - test_ruler[b]
                        if d in diffs_set:
                            ok = False
                            break
                        diffs_set.add(d)
                    if not ok:
                        break
                
                if ok:
                    # Compute new score
                    new_score = compute_score_golomb(test_ruler)
                    if new_score > best_score_local:
                        best_score_local = new_score
                        best_ruler = list(test_ruler)
                        current = list(test_ruler)
                        found_move = True
                        break
            if found_move:
                continue
            
            # Try shifting right
            for delta in range(1, 10):
                new_val = old_val + delta
                if new_val in current:
                    continue
                
                test_ruler = list(current)
                test_ruler[idx] = new_val
                test_ruler.sort()
                
                diffs_set = set()
                ok = True
                for a in range(n):
                    for b in range(a):
                        d = test_ruler[a] - test_ruler[b]
                        if d in diffs_set:
                            ok = False
                            break
                        diffs_set.add(d)
                    if not ok:
                        break
                
                if ok:
                    new_score = compute_score_golomb(test_ruler)
                    if new_score > best_score_local:
                        best_score_local = new_score
                        best_ruler = list(test_ruler)
                        current = list(test_ruler)
                        found_move = True
                        break
            if found_move:
                continue
                
            # If no move found, try a larger random jump
            # Or swap two pins
            if random.random() < 0.1:
                idx2 = random.randint(1, n - 1)
                if idx2 != idx:
                    test_ruler = list(current)
                    test_ruler[idx], test_ruler[idx2] = test_ruler[idx2], test_ruler[idx]
                    test_ruler.sort()
                    
                    diffs_set = set()
                    ok = True
                    for a in range(n):
                        for b in range(a):
                            d = test_ruler[a] - test_ruler[b]
                            if d in diffs_set:
                                ok = False
                                break
                            diffs_set.add(d)
                        if not ok:
                            break
                    
                    if ok:
                        new_score = compute_score_golomb(test_ruler)
                        if new_score > best_score_local:
                            best_score_local = new_score
                            best_ruler = list(test_ruler)
                            current = list(test_ruler)
                            found_move = True
            if found_move:
                continue

        return best_ruler, best_score_local

    # Main search loop
    # Try different sizes n
    max_n_to_try = 12
    
    for n in range(3, max_n_to_try + 1):
        if time.time() > deadline - 5:
            break
            
        # Get an initial Golomb ruler
        initial_ruler = greedy_golomb(n, max_span=2000)
        
        if not initial_ruler:
            # Try a different greedy strategy or random restart
            found = False
            for _ in range(10):
                random.seed(n * 100 + random.randint(0, 100))
                initial_ruler = greedy_golomb(n, max_span=3000)
                if initial_ruler:
                    found = True
                    break
            if not found:
                continue
                
        # Optimize the ruler
        optimized_ruler, score = optimize_golomb(initial_ruler, max_iters=2000)
        
        if score > best_score:
            best_score = score
            best_set = list(optimized_ruler)

    # Also try some known optimal rulers for small n
    known_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 18, 23, 31, 42],
        [0, 1, 4, 10, 18, 23, 31, 42, 52],
    ]
    
    for ruler in known_rulers:
        s = compute_score_golomb(ruler)
        if s > best_score:
            best_score = s
            best_set = list(ruler)
        
        # Try to improve known rulers too
        if time.time() < deadline - 10:
            opt_ruler, opt_score = optimize_golomb(ruler, max_iters=1000)
            if opt_score > best_score:
                best_score = opt_score
                best_set = list(opt_ruler)

    return best_set

# EVOLVE-BLOCK-END
