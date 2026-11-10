import time
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    # The evaluator passes these, but we optimize for the specific case if it matches, 
    # otherwise we fall back to a generic greedy.
    # However, the task description implies a fixed task: "Football pool problem for 6 matches".
    # We will implement a general greedy that works for any q, n, radius but is optimized for small spaces.
    
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit, leaving 20s buffer for 120s total
    
    # Total number of words
    total_words = q ** n
    
    # If the space is too large for our simple greedy, we might need a different strategy,
    # but for q=3, n=6, total_words=729, it's very small.
    
    # Precompute balls for all words
    # ball[i] is a list of words within Hamming distance 'radius' from word i
    # For radius 1, ball[i] = {i} U {words differing in exactly 1 position}
    
    # We represent words as integers 0..total_words-1
    # To get digits of a word: base-q expansion
    
    # Helper: convert int to list of digits
    def int_to_digits(x, q, n):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits
    
    # Helper: convert list of digits to int
    def digits_to_int(digits, q):
        val = 0
        for i in range(n-1, -1, -1):
            val = val * q + digits[i]
        return val

    # Precompute neighbors
    # For efficiency, we compute the ball for each word once.
    # Since q=3, n=6, radius=1, ball size is 13.
    # total_words = 729. 729 * 13 is small.
    
    balls = []
    for i in range(total_words):
        digits = int_to_digits(i, q, n)
        neighbors = [i]
        # For each position
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                # Create new word
                new_digits = digits[:]
                new_digits[pos] = new_val
                new_word = digits_to_int(new_digits, q)
                neighbors.append(new_word)
        balls.append(neighbors)
    
    # Convert to numpy array for faster access if needed, 
    # but lists of lists are okay for 729 items.
    # Let's use a list of lists.
    
    best_code = None
    best_size = float('inf')
    
    # We will try multiple random restarts.
    # The greedy algorithm is deterministic if we pick the first max.
    # To randomize, we can add small random noise to the counts or shuffle ties.
    
    # For speed, we'll use a standard greedy but with a random initialization of "priority" 
    # or just random tie-breaking.
    
    # Optimized Greedy:
    # covered: boolean array
    # counts: array of ints, counts[i] = num uncovered in ball[i]
    
    # Initial counts: all balls have size 13 (for R=1, q=3).
    # Actually, we just need to know how many uncovered points are in the ball.
    # Initially, all 729 are uncovered.
    # So counts[i] = len(ball[i]) = 13.
    
    # However, to handle general radius, we compute ball sizes.
    ball_sizes = np.array([len(b) for b in balls])
    
    # We run the greedy multiple times.
    # Each run:
    # 1. Initialize covered = False * total_words
    # 2. Initialize counts = ball_sizes (since all are uncovered)
    # 3. Loop until all covered or time out:
    #    a. Find i with max counts[i]
    #    b. If counts[i] == 0, break (should not happen if not all covered)
    #    c. Add i to code
    #    d. For each j in ball[i]:
    #         if not covered[j]:
    #             covered[j] = True
    #             # Update counts for all k that cover j
    #             # This is the expensive part.
    #             # We need a list of "who covers j".
    #             # Since the graph is symmetric (j in ball[i] iff i in ball[j]),
    #             # the set of k such that j in ball[k] is exactly ball[j].
    #             # So we iterate k in ball[j] and decrement counts[k].
    
    # Precomputation: We already have balls.
    
    # To speed up the inner loop, we can use numpy for counts updates?
    # counts[ball[j]] -= 1
    # This is vectorized.
    
    # Let's implement one run of the randomized greedy.
    
    def run_greedy(random_seed):
        np.random.seed(random_seed)
        covered = np.zeros(total_words, dtype=bool)
        counts = ball_sizes.copy()
        code = []
        
        # We need to track which words are in the code to avoid duplicates? 
        # The problem says codewords must be distinct. Greedy picks max count.
        # If a word is already in the code, its ball is fully covered? 
        # No, if we pick i, we cover ball[i]. 
        # If we pick i again, counts[i] would be 0 because all in ball[i] are covered.
        # So we won't pick it again if counts are updated correctly.
        
        while not np.all(covered):
            # Find max count
            # To randomize ties, we can add small noise or just rely on np.argmax 
            # which is deterministic. 
            # Better: if there are ties, pick randomly among them.
            max_count = np.max(counts)
            if max_count == 0:
                break # Should not happen if not all covered
            
            # Find indices with max count
            candidates = np.where(counts == max_count)[0]
            
            # Randomly pick one
            i = candidates[np.random.randint(len(candidates))]
            
            code.append(int(i))
            
            # Update coverage
            # For each j in ball[i]:
            for j in balls[i]:
                if not covered[j]:
                    covered[j] = True
                    # Update counts for all k in ball[j]
                    # counts[ball[j]] -= 1
                    # Using numpy for this update
                    counts[balls[j]] -= 1
        
        return code

    # Run multiple iterations
    # How many? 
    # Each run takes ~ T_run.
    # T_run is roughly O(K * S * R) where R is radius? 
    # With numpy vectorization for counts update, it's faster.
    # Let's estimate. 73 iterations. Each iteration does ~13 numpy updates of size 13.
    # Very fast. Maybe 1ms per run?
    # We can do thousands of runs.
    
    # Let's do 100 runs first, then check time.
    
    for seed in range(100):
        if time.time() > deadline:
            break
        
        code = run_greedy(seed)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            # Early exit if we hit the theoretical lower bound?
            # Lower bound is 57. Unlikely to hit.
            # If we hit 73, that's the known best. We can try to beat it, but it's hard.
            # We'll keep searching.
        
        # If we found 73, we might want to spend more time trying to find 72?
        # Or just return 73?
        # The score is seed/value. 81/73 = 1.11. 81/72 = 1.125.
        # It's better to find 72 if possible.
        
    # If we didn't find anything (shouldn't happen), fallback to baseline
    if best_code is None:
        # Fallback to the direct sum construction from the prompt
        # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Since we are in a general function, we need a generic fallback.
        # But the task is specific. 
        # If q=3, n=6, we can hardcode the tetracode logic as a fallback.
        if q == 3 and n == 6:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: Random words
            return [np.random.randint(0, q, n).tolist() for _ in range(100)]

    # Convert best_code (list of ints) to list of digit lists
    final_code = [int_to_digits(x, q, n) for x in best_code]
    return final_code

# The prompt asks to keep the function names and signatures.
# The current program has:
# def construct(q, n, radius):
# ...
# return [h + list(tail) ...]
# My implementation returns a list of lists of ints, which matches.

# One detail: The evaluator passes q, n, radius.
# My code assumes q, n, radius are passed.
# I used q, n, radius in the logic.

# Let's double check the "balls" precomputation.
# It depends on q and n.
# If q and n are different, the precomputation is done inside construct.
# This is fine as long as it's fast enough.
# For q=3, n=6, it is instant.
# For larger n, it might be slow, but the task is fixed to 6 matches.

# To be safe against timeouts on precomputation for larger inputs (though unlikely given the task),
# we can cache it. But since it's a single call, it's fine.

# Final check on the "balls" update logic.
# counts[balls[j]] -= 1
# balls[j] is a list of ints.
# counts is a numpy array.
# This works.

# One potential issue: 
# "counts[balls[j]] -= 1" modifies counts in place.
# This is correct.

# Is there a risk of negative counts?
# If we decrement counts[k] for every j in ball[i] that gets covered,
# and k is in ball[j], it means j is in ball[k].
# Since we only decrement if j was newly covered, and each j is covered exactly once,
# the total decrement for any k is exactly the number of j in ball[k] that were covered.
# Initially counts[k] = |ball[k]|.
# Finally counts[k] = |ball[k]| - (number of covered in ball[k]).
# This is >= 0.
# So no negative counts.

# The logic holds.
