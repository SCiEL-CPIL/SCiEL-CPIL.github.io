import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate initial code
    # Direct sum of tetracode (len 4) and all tails (len n-4)
    if n == 4:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        code = [tuple(t) for t in tetracode]
    else:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n-4):
                code.append(tuple(h + list(tail)))
    
    # If n < 4, we just take all words? No, radius 1.
    # For general q, n, radius, we need a general initializer.
    # But the task is specific to q=3, n=6, radius=1 mostly.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies q=3, n=6, radius=1 is the main case, but we should be robust.
    # However, the "best known" and "lower bound" are specific to 3,6,1.
    # Let's assume the main case is 3,6,1.
    
    if q == 3 and n == 6 and radius == 1:
        # Optimization specific to this case
        pass

    # General fallback initializer: Greedy construction
    # But the seed is 81. If we start from scratch, we might not beat 81 quickly.
    # The direct sum is a good structured start.
    
    # Let's stick to the direct sum start for q=3, n>=4.
    # If n < 4, we can just use all words (size 3^n) and remove.
    # For n=6, q=3, size is 81.
    
    # Helper to check coverage
    space_size = q ** n
    
    def get_ball(c):
        # Yields all words within radius R of c
        # For R=1, it's c and its n(q-1) neighbors
        yield c
        for i in range(n):
            orig = c[i]
            for v in range(q):
                if v == orig: continue
                # Create new word
                # Tuples are immutable, so we convert
                # Optimization: precompute neighbors?
                # For n=6, q=3, R=1, ball size is 13.
                # 81 codewords * 13 = 1053 words to check per pass.
                # We can just generate them.
                nc = list(c)
                nc[i] = v
                yield tuple(nc)

    def compute_counts(code_list):
        counts = [0] * space_size
        for c in code_list:
            # Convert c to index
            # c is a tuple of ints
            # Index = sum(c[i] * q^(n-1-i))
            idx = 0
            power = 1
            for i in range(n-1, -1, -1):
                idx += c[i] * power
                power *= q
            
            # Add 1 to all words in ball(c)
            # We need to map word -> index
            # It's faster to precompute the indices of the ball for each codeword?
            # Or just compute on the fly.
            # Given the small size, on the fly is fine.
            
            for w in get_ball(c):
                w_idx = 0
                power = 1
                for i in range(n-1, -1, -1):
                    w_idx += w[i] * power
                    power *= q
                counts[w_idx] += 1
        return counts

    # To speed up index calculation, we can precompute a mapping from tuple to index
    # Or just use the tuple as a key in a dictionary?
    # List of size 729 is fast.
    
    # Let's optimize the count calculation.
    # Precompute ball indices for all possible codewords? No, codewords change.
    # But we can precompute the "neighbors" structure.
    
    # Let's use a dictionary for counts if space is large, but here 729 is small.
    # List is faster.
    
    # We need a fast way to get index of a word.
    def word_to_idx(w):
        idx = 0
        p = 1
        for i in range(n-1, -1, -1):
            idx += w[i] * p
            p *= q
        return idx

    # Precompute indices for all words in the space?
    # There are 729 words.
    # We can create a list of all words and their indices.
    all_words = []
    word_indices = {}
    for w in itertools.product(range(q), repeat=n):
        w_t = tuple(w)
        idx = word_to_idx(w_t)
        all_words.append(w_t)
        word_indices[w_t] = idx
    
    # Precompute balls for all words in the space
    # ball_indices[word] = list of indices of words in ball(word)
    ball_indices = {}
    for w in all_words:
        b = list(get_ball(w))
        ball_indices[w] = [word_indices[x] for x in b]

    # Now, compute_counts can be faster if we use the precomputed balls
    # But we still iterate over codewords.
    
    def compute_counts_fast(code_list):
        counts = [0] * space_size
        for c in code_list:
            c_idx = word_indices[c]
            # The ball of c is the same as the ball of any word? No.
            # But we precomputed ball_indices for all words.
            # So for codeword c, we just add 1 to counts[i] for i in ball_indices[c]
            for idx in ball_indices[c]:
                counts[idx] += 1
        return counts

    # Check removability
    def find_removable(code_list, counts):
        removable = []
        for c in code_list:
            # Check if all words in ball(c) have count >= 2
            # If so, removing c leaves them with count >= 1
            can_remove = True
            for idx in ball_indices[c]:
                if counts[idx] < 2:
                    can_remove = False
                    break
            if can_remove:
                removable.append(c)
        return removable

    # Main loop
    best_code = list(code)
    
    while True:
        if time.time() > deadline:
            break
        
        counts = compute_counts_fast(best_code)
        removable = find_removable(best_code, counts)
        
        if not removable:
            # Try a few random restarts or perturbations?
            # Or just break.
            # Let's try a simple perturbation: remove a random non-removable, add a random word?
            # This is risky and slow.
            # For 3,6,1, greedy usually gets to 73 or 74.
            # Let's break and return best.
            break
        else:
            # Remove one
            # To be deterministic-ish but effective, pick one with min "redundancy" or just random
            # Random might help escape local minima if we had them, but here we are just pruning.
            # Let's pick the first one for speed, or random for variety.
            c_to_remove = random.choice(removable)
            best_code.remove(c_to_remove)
            
    return [list(c) for c in best_code]
