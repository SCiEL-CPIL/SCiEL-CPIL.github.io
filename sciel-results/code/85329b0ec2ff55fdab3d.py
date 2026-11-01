import time
import random
import sys

# Constants
Q = 3
N = 7
RADIUS = 1
TOTAL_WORDS = 3**N  # 2187

# Precompute balls
# A word is represented by an integer 0..2186.
# To check distance, we need to compare digits.
# It's faster to precompute the list of indices covered by each word.
# Then convert those lists to integer masks.

def decode_word(index):
    """Convert integer index to list of digits."""
    digits = []
    for _ in range(N):
        digits.append(index % Q)
        index //= Q
    return digits

def encode_word(digits):
    """Convert list of digits to integer index."""
    val = 0
    for i in range(N-1, -1, -1):
        val = val * Q + digits[i]
    return val

# Precompute neighbors for each word
# balls[i] will be an integer bitmask where bit j is 1 if word j is within distance 1 of word i.
balls = [0] * TOTAL_WORDS

# Generate all words
words = [decode_word(i) for i in range(TOTAL_WORDS)]

for i in range(TOTAL_WORDS):
    w = words[i]
    mask = 1 << i  # The word itself
    
    # Neighbors: change one position to the other 2 values
    for pos in range(N):
        orig_val = w[pos]
        for new_val in range(Q):
            if new_val == orig_val:
                continue
            # Create new word
            new_w = w[:]
            new_w[pos] = new_val
            j = encode_word(new_w)
            mask |= (1 << j)
    balls[i] = mask

# Full mask of all words
FULL_MASK = (1 << TOTAL_WORDS) - 1

def greedy_construct(random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    uncovered = FULL_MASK
    code = []
    
    # To speed up, we can just iterate through all candidates each time.
    # 2187 candidates * ~200 steps * bit_count is fast enough in Python?
    # 2187 * 200 = 437,400 operations. Each is a big int & and bit_count.
    # This is likely fast enough for one run.
    
    while uncovered:
        best_idx = -1
        best_count = -1
        best_candidates = []
        
        for i in range(TOTAL_WORDS):
            # Only consider if i is not already in code? 
            # Actually, adding a duplicate is useless. 
            # We can skip if i is in code.
            # But checking `if i in code` is O(N_code). 
            # Better to just allow it, but if it's in code, its new coverage is 0.
            # So we can just compute coverage.
            
            # Optimization: If i is already in code, coverage is 0.
            # We can maintain a set of used indices.
            # But for simplicity and speed of inner loop, let's just compute.
            # Actually, if i is in code, balls[i] & uncovered will be 0 because uncovered has 0 where balls[i] has 1.
            # So we don't need to explicitly skip.
            
            cov = (balls[i] & uncovered).bit_count()
            if cov > best_count:
                best_count = cov
                best_idx = i
                best_candidates = [i]
            elif cov == best_count and cov > 0:
                best_candidates.append(i)
        
        if best_count == 0:
            # This shouldn't happen if we have a valid cover, but if we stuck, break
            break
            
        # Pick randomly from best candidates to introduce variation
        selected = random.choice(best_candidates)
        code.append(selected)
        uncovered &= ~balls[selected]
        
    return code

def reduce_code(code):
    """Remove redundant codewords."""
    if not code:
        return code
    
    # counts[j] = number of codewords in code that cover word j
    counts = [0] * TOTAL_WORDS
    for i in code:
        mask = balls[i]
        # Iterate through set bits in mask?
        # Or just loop through all words? 2187 * len(code) might be slow if done naively.
        # But len(code) is ~200. 200 * 2187 = 437k. 
        # Inside, we need to increment counts for bits in mask.
        # Efficient way: while mask: lowbit = mask & -mask; idx = lowbit.bit_length()-1; counts[idx]+=1; mask ^= lowbit
        m = mask
        while m:
            lowbit = m & -m
            idx = lowbit.bit_length() - 1
            counts[idx] += 1
            m ^= lowbit
            
    new_code = []
    for i in code:
        # Check if i is redundant
        # i is redundant if for all j in balls[i], counts[j] > 1
        redundant = True
        m = balls[i]
        while m:
            lowbit = m & -m
            idx = lowbit.bit_length() - 1
            if counts[idx] == 1:
                redundant = False
                break
            m ^= lowbit
        
        if redundant:
            # Remove i
            # Decrement counts
            m = balls[i]
            while m:
                lowbit = m & -m
                idx = lowbit.bit_length() - 1
                counts[idx] -= 1
                m ^= lowbit
        else:
            new_code.append(i)
            
    # We can repeat this reduction until no more changes
    # But one pass is often enough to remove obvious redundancies.
    # Let's do a second pass if size changed significantly?
    # For now, single pass.
    
    return new_code

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    best_code = None
    best_size = float('inf')
    
    # Initial run with fixed seed to ensure we have a valid code quickly
    code = greedy_construct(0)
    code = reduce_code(code)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code
        
    # Random restarts
    seed = 1
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # If we hit 186, we can stop early? 
            # 186 is the best known. Let's aim for it.
            if best_size <= 186:
                break
        seed += 1
        
    # Convert indices to lists of digits
    return [words[i] for i in best_code]

if __name__ == "__main__":
    # Self-test
    res = construct(3, 7, 1)
    print(f"Size: {len(res)}")
    # Verify validity
    covered = 0
    for i in range(len(res)):
        covered |= balls[encode_word(res[i])]
    if covered == FULL_MASK:
        print("Valid")
    else:
        print("Invalid")
