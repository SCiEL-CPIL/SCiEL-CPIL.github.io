import time
import random
import itertools

# Precomputation for q=3, n=6, radius=1
Q = 3
N = 6
R = 1
TOTAL_WORDS = Q ** N  # 729

# Generate all words and map to index
words = []
word_to_idx = {}
for i in range(TOTAL_WORDS):
    # Convert i to base-3 representation of length 6
    w = []
    val = i
    for _ in range(N):
        w.append(val % Q)
        val //= Q
    w = tuple(w)
    words.append(w)
    word_to_idx[w] = i

# Precompute balls as bitmasks
# ball[i] is a bitmask of all words within distance R from words[i]
balls = [0] * TOTAL_WORDS

def hamming_distance(w1, w2):
    return sum(1 for a, b in zip(w1, w2) if a != b)

for i in range(TOTAL_WORDS):
    mask = 1 << i
    # Include self
    # Include neighbors
    for j in range(i + 1, TOTAL_WORDS):
        if hamming_distance(words[i], words[j]) <= R:
            mask |= 1 << j
            balls[j] |= 1 << i # Symmetric
    balls[i] = mask

# AllMask for checking full coverage
ALL_MASK = (1 << TOTAL_WORDS) - 1

def construct(q, n, radius):
    # Fallback for other parameters
    if not (q == 3 and n == 6 and radius == 1):
        # Simple greedy for other cases
        total = q ** n
        # This is a naive fallback, might be slow or large
        # But the task specifically targets the 3,6,1 case
        # We'll implement a basic greedy here
        words_list = list(itertools.product(range(q), repeat=n))
        word_to_idx_gen = {w: i for i, w in enumerate(words_list)}
        
        def dist(w1, w2):
            return sum(1 for a,b in zip(w1,w2) if a!=b)
            
        balls_gen = []
        for i, w in enumerate(words_list):
            mask = 0
            for j, w2 in enumerate(words_list):
                if dist(w, w2) <= radius:
                    mask |= 1 << j
            balls_gen.append(mask)
        
        all_mask_gen = (1 << total) - 1
        
        code = []
        counts = [0] * total
        covered_mask = 0
        
        # Greedy add
        while covered_mask != all_mask_gen:
            best_c = -1
            best_gain = -1
            # Sample or iterate all? For small sizes iterate all.
            # For large sizes, this will be slow. 
            # Given the specific task is 3,6,1, this fallback is rarely used.
            # We'll just do a simple random greedy for fallback.
            candidates = [i for i in range(total) if i not in code]
            random.shuffle(candidates)
            for c in candidates:
                gain = (balls_gen[c] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                if best_gain == 0:
                    break
            if best_c == -1:
                break
            code.append(best_c)
            covered_mask |= balls_gen[best_c]
            for w in range(total):
                if (balls_gen[best_c] >> w) & 1:
                    counts[w] += 1
            
        return [list(words_list[i]) for i in code]

    # Specific optimization for q=3, n=6, radius=1
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    # Use multiple restarts
    while time.time() - start_time < time_limit:
        # Initialize with random set of codewords
        # Start size around 80
        init_size = 80
        code = random.sample(range(TOTAL_WORDS), init_size)
        counts = [0] * TOTAL_WORDS
        covered_mask = 0
        
        for c in code:
            covered_mask |= balls[c]
            for w in range(TOTAL_WORDS):
                if (balls[c] >> w) & 1:
                    counts[w] += 1
        
        # Greedy completion
        while covered_mask != ALL_MASK:
            best_c = -1
            best_gain = -1
            # Find codeword covering most uncovered words
            # Iterate all 729 is fast enough
            for c in range(TOTAL_WORDS):
                if c in code:
                    continue
                gain = (balls[c] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                if best_gain == 0:
                    break
            if best_c == -1:
                break
            code.append(best_c)
            covered_mask |= balls[best_c]
            for w in range(TOTAL_WORDS):
                if (balls[best_c] >> w) & 1:
                    counts[w] += 1
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
        
        # Local Search: Hill Climbing
        # We try to reduce the size of the code
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # 1. Try to remove redundant codewords
            # A codeword c is redundant if all words in its ball are covered by others
            # i.e., for all w in ball[c], counts[w] >= 2
            removal_idx = -1
            for i, c in enumerate(code):
                # Check if any word in ball[c] has count 1
                # If all have count >= 2, it's removable
                removable = True
                mask = balls[c]
                # Iterate set bits in mask
                m = mask
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if counts[w] < 2:
                        removable = False
                        break
                    m ^= lsb
                if removable:
                    removal_idx = i
                    break
            
            if removal_idx != -1:
                c = code.pop(removal_idx)
                # Update counts
                mask = balls[c]
                m = mask
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    counts[w] -= 1
                    m ^= lsb
                improved = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                continue # Restart loop to check for more removals
            
            # 2. Try to swap
            # If no removal possible, try to swap a codeword to enable future removals
            # Pick a random codeword c from code
            if len(code) > 0:
                i = random.randint(0, len(code) - 1)
                c = code[i]
                # Pick a random candidate c' not in code
                # To speed up, we can just pick random and check membership
                # Or maintain a set of code
                code_set = set(code)
                c_new = random.randint(0, TOTAL_WORDS - 1)
                if c_new in code_set:
                    continue
                
                # Check if swap is valid
                # Swap is valid if replacing c with c_new doesn't leave any word uncovered
                # This happens if:
                # For all w in ball[c] \ ball[c_new], counts[w] >= 2
                # (Words in ball[c_new] \ ball[c] are gained, so no problem)
                # (Words in intersection stay same)
                # (Words outside both stay same)
                
                # We need to check words that are in ball[c] but NOT in ball[c_new]
                # AND have count 1. If any such word exists, swap is invalid.
                
                mask_c = balls[c]
                mask_c_new = balls[c_new]
                
                # Words in ball[c] but not ball[c_new]
                # We iterate over bits in mask_c
                valid_swap = True
                m = mask_c
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    # Check if w is in mask_c_new
                    if not (mask_c_new & lsb):
                        # w is in ball[c] but not ball[c_new]
                        if counts[w] < 2:
                            valid_swap = False
                            break
                    m ^= lsb
                
                if valid_swap:
                    # Perform swap
                    code[i] = c_new
                    code_set.add(c_new)
                    code_set.remove(c)
                    
                    # Update counts
                    # Subtract counts for c
                    m = mask_c
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        counts[w] -= 1
                        m ^= lsb
                    # Add counts for c_new
                    m = mask_c_new
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        counts[w] += 1
                        m ^= lsb
                    
                    # Set improved to True to allow removal check in next iteration
                    # The swap itself doesn't reduce size, but it might make
                    # another codeword redundant.
                    improved = True
                    # Note: We don't update best_size here as size didn't change
                    # But we continue the loop to try removals.
                    # To avoid infinite loops of swaps without removal,
                    # we could limit consecutive swaps, but usually removals happen.
                    # If stuck, we might need to revert or randomize more.
                    # For now, let's rely on the random nature.
                    # If we swap back and forth, we might loop.
                    # To prevent this, we can keep a history or just accept that
                    # we might waste some time. Given 110s, it's okay.
                    # However, if we swap c->c_new and then c_new->c, we loop.
                    # We can check if c_new is "better" in some metric?
                    # Simpler: Just accept. If it loops, we'll hit time limit.
                    # But we want to find size 73.
                    # Let's add a small penalty: if we swap, we prioritize removals next.
                    # The `improved=True` ensures we go back to the top and try removals first.
                    # If no removal, we try another swap.
                    # This should work.

    if best_code is None:
        # Fallback to the original construction if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(words[i]) for i in best_code]
