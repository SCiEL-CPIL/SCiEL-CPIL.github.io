"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def construct(q, n, radius):
    start_time = time.time()
    time_limit = 145.0 # 5s safety margin
    
    # Handle the specific case of the task: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        # 1. Generate initial 81 codewords using Tetracode product
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
        
        # Helper functions for integer representation
        def to_int(word):
            val = 0
            for d in word:
                val = val * 3 + d
            return val
        
        def from_int(val, n):
            word = [0] * n
            for i in range(n - 1, -1, -1):
                word[i] = val % 3
                val //= 3
            return word

        init_ints = [to_int(c) for c in init_codewords]
        
        # Precompute coverage balls for all 729 possible words
        # ball[w] is a list of words within distance 1 of w
        all_balls = []
        for w in range(3**n):
            ball = [w]
            word = from_int(w, n)
            for pos in range(n):
                orig_val = word[pos]
                for d in range(3):
                    if d != orig_val:
                        new_word = word[:]
                        new_word[pos] = d
                        ball.append(to_int(new_word))
            all_balls.append(ball)
            
        best_codewords = init_codewords
        best_size = len(best_codewords)
        
        # Greedy reduction function
        def greedy_reduce(initial_ints):
            current_ints = list(initial_ints)
            counts = [0] * (3**n)
            
            # Initialize coverage counts
            for c in current_ints:
                for w in all_balls[c]:
                    counts[w] += 1
            
            # Iteratively remove redundant codewords
            changed = True
            while changed:
                changed = False
                for i in range(len(current_ints)):
                    c = current_ints[i]
                    # Check if all words covered by c are covered by at least one other codeword
                    removable = True
                    for w in all_balls[c]:
                        if counts[w] <= 1:
                            removable = False
                            break
                    
                    if removable:
                        # Remove c
                        for w in all_balls[c]:
                            counts[w] -= 1
                        current_ints.pop(i)
                        changed = True
                        # Restart scan to check new removabilities
                        break
                        
            return current_ints

        # Run multiple random restarts to escape local minima
        while time.time() - start_time < time_limit:
            shuffled = list(init_ints)
            random.shuffle(shuffled)
            
            result = greedy_reduce(shuffled)
            
            if len(result) < best_size:
                best_size = len(result)
                best_codewords = [from_int(c, n) for c in result]
                
            # The best known lower bound is 71, and 73 is the best known upper bound.
            # If we find 73, it is highly likely to be optimal or near-optimal.
            # We can stop early to save time, but continuing might find 72 (unlikely) or confirm 73.
            # Given the search space, 73 is the target.
            if best_size == 73:
                # We found a very good solution. We can break to ensure we return quickly.
                # However, if we have lots of time, we could keep searching for 72.
                # But 72 is not known to exist. 71 is the lower bound.
                # Let's run a bit more just in case, but 73 is the practical target.
                # Actually, breaking is safer to avoid timeout if random restarts are slow.
                # But they are fast. Let's just break if we hit 73 to be safe and fast.
                break

        return best_codewords

    # Fallback for other parameters
    # Use the product construction
    if n <= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        if n == 3:
            return [t[:3] for t in tetracode]
        if n == 2:
            return [[a, b] for a in range(3) for b in range(3)]
        if n == 1:
            return [[0], [1], [2]]
        
    # For n > 4, use product of tetracode and full space of remaining coords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # If n < 4 and q != 3 or radius != 1, just return all words
        return [list(t) for t in itertools.product(range(q), repeat=n)]

# EVOLVE-BLOCK-END
