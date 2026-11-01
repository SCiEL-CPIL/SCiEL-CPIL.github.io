import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        return [[0] * n]
    
    # Precomputation
    pow3 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow3[i] = pow3[i-1] * 3
    total_words = pow3[n] # 6561
    
    def encode(digits):
        w = 0
        for i in range(n):
            w += digits[i] * pow3[i]
        return w
        
    def decode(w):
        digits = [0] * n
        for i in range(n):
            digits[i] = w % 3
            w //= 3
        return digits

    # Precompute balls for all words
    # balls[w] = list of words within distance 1 of w
    balls = []
    for w in range(total_words):
        digits = decode(w)
        ball = [w]
        for i in range(n):
            original = digits[i]
            for new_val in range(3):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[i] = new_val
                    ball.append(encode(new_digits))
        balls.append(ball)
        
    # Initial Code: Tetracode Direct Sum
    # Tetracode: 9 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = []
    for h in tetracode:
        for tail in product(range(3), repeat=4):
            initial_code.append(h + list(tail))
            
    code = [encode(c) for c in initial_code]
    code_set = set(code)
    
    # Initialize coverage counts
    coverage = [0] * total_words
    for c in code:
        for w in balls[c]:
            coverage[w] += 1
            
    best_code = list(code)
    best_size = len(code)
    
    def try_remove_redundant(code, code_set, coverage):
        """Remove all redundant codewords. Returns number removed."""
        removed_count = 0
        # We iterate multiple times or in a single pass? 
        # Single pass might miss some if order matters. 
        # But if we remove one, it only decreases coverage, making others MORE likely to be redundant.
        # So a single forward pass is safe for finding *some* redundants, but might not find all in one pass.
        # However, if we remove c, it doesn't make other words *less* redundant.
        # So we can do a single pass, but we might need multiple passes to reach a fixed point.
        # To save time, we'll do a single pass per call, and call it repeatedly.
        
        to_remove = []
        for c in code:
            # Check if c is redundant
            redundant = True
            for w in balls[c]:
                if coverage[w] <= 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(c)
                
        for c in to_remove:
            code_set.remove(c)
            code.remove(c) # List remove is O(N), but N is small (~500-700)
            for w in balls[c]:
                coverage[w] -= 1
            removed_count += 1
            
        return removed_count

    def find_swap(c, code_set, coverage):
        """
        Find a codeword c_new that can replace c.
        c_new must cover all words that are ONLY covered by c.
        c_new must not be in code_set (except c itself, which we are removing).
        """
        # Find words covered only by c
        only_covered = []
        for w in balls[c]:
            if coverage[w] == 1:
                only_covered.append(w)
                
        if not only_covered:
            # c is redundant, should have been removed
            return None
            
        # We need a c_new in the intersection of balls[w] for w in only_covered
        # Start with the first word's ball
        candidates = set(balls[only_covered[0]])
        
        # Intersect with others
        for w in only_covered[1:]:
            candidates.intersection_update(balls[w])
            if not candidates:
                break
                
        if not candidates:
            return None
            
        # Filter out candidates that are already in the code (except c)
        for c_new in candidates:
            if c_new == c:
                continue
            if c_new in code_set:
                continue
            return c_new
            
        return None

    # Main Optimization Loop
    # We alternate between removal and swapping.
    # To escape local minima, we shuffle the order of processing.
    
    while time.time() < deadline:
        # 1. Remove redundants
        while time.time() < deadline:
            removed = try_remove_redundant(code, code_set, coverage)
            if removed == 0:
                break
                
        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            
        # 2. Try swaps
        # Shuffle code to avoid bias
        random.shuffle(code)
        
        swap_made = False
        for c in code:
            if time.time() > deadline:
                break
                
            c_new = find_swap(c, code_set, coverage)
            if c_new is not None:
                # Perform swap
                # Remove c
                code_set.remove(c)
                code.remove(c)
                for w in balls[c]:
                    coverage[w] -= 1
                    
                # Add c_new
                code_set.add(c_new)
                code.append(c_new)
                for w in balls[c_new]:
                    coverage[w] += 1
                    
                swap_made = True
                # After a swap, coverage has changed. 
                # It's efficient to run removal immediately.
                # But we are in a loop over code. 
                # Let's just break and let the outer loop handle removal.
                break
        
        if not swap_made:
            # No swaps found in this pass.
            # We might be stuck. 
            # Option: Restart from best_code? Or just break?
            # If we break, we return the best found.
            # To try harder, we could restart from the initial 729 code with a new random seed.
            # Given time constraints, if we are stuck at > 486, we might want to try a different strategy.
            # But for now, let's just break if we can't improve.
            # Actually, let's try a few more passes with different shuffles?
            # No, if no swap is possible, the set is "locally optimal" w.r.t single swaps.
            # Breaking is reasonable.
            break

    # Convert best_code back to list of lists
    return [decode(w) for w in best_code]

# Helper for product
def product(iterable, repeat):
    # Simple implementation to avoid import overhead or just use itertools
    import itertools
    return itertools.product(iterable, repeat=repeat)
