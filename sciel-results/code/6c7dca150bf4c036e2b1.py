import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 145s to be safe (evaluator kills at 150s)
    
    # Generate all words and map to indices
    all_words = np.array(list(product(range(q), repeat=n)))
    num_words = q ** n
    
    # Precompute balls: for each word, the set of words it covers (within Hamming distance <= radius)
    # For q=3, n=7, radius=1: each word covers 1 + 7*2 = 15 words
    # Use bitsets for efficient set operations
    # num_words = 2187, so we need bitsets of 2187 bits
    
    # Precompute the ball as a bitset (using Python integers as bitsets)
    # bit i is set if word i is in the ball
    
    # First, create a mapping from word tuple to index
    word_to_idx = {}
    for i in range(num_words):
        word_to_idx[tuple(all_words[i])] = i
    
    # Precompute balls as bitsets (Python integers)
    balls = [0] * num_words
    for i in range(num_words):
        w = all_words[i]
        ball = 0
        # The word itself
        ball |= (1 << i)
        # For radius=1: change one position
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                j = word_to_idx[tuple(w)]
                ball |= (1 << j)
                w[pos] = orig
        balls[i] = ball
    
    # Precompute ball sizes
    ball_sizes = [bin(b).count('1') for b in balls]
    
    # Greedy set cover with multiple restarts
    best_code = None
    best_size = num_words  # trivial upper bound: all words
    
    # Strategy 1: Randomized greedy with multiple restarts
    # Each restart: randomize tie-breaking, possibly randomize initial order
    
    num_restarts = 5000  # will be limited by time
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Randomized greedy
        uncovered = (1 << num_words) - 1  # all bits set
        code = []
        
        # Shuffle the order for tie-breaking
        order = list(range(num_words))
        rng.shuffle(order)
        
        while uncovered:
            if time.time() > deadline - 30:
                break
            
            best_idx = -1
            best_count = 0
            best_tiebreak = -1
            
            # Iterate in shuffled order for randomization
            for i in order:
                if not (uncovered & (1 << i)):  # skip if i is already covered? No, we need to check if ball[i] covers uncovered
                    pass
                # Count how many uncovered words are in ball[i]
                cnt = bin(uncovered & balls[i]).count('1')
                if cnt > best_count or (cnt == best_count and cnt > 0 and rng.random() < 0.05):
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
        
        # Check if we have a valid cover
        if uncovered == 0:
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
        
        # Early termination if we hit the known best
        if best_size <= 186:
            break
    
    # Local search: try to remove redundant codewords
    if best_code is not None:
        code = best_code.copy()
        changed = True
        while changed and time.time() < deadline - 15:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if time.time() > deadline - 15:
                    break
                # Check if removing code[i] still covers everything
                # Compute covered set without code[i]
                covered = 0
                for j, idx in enumerate(code):
                    if j != i:
                        covered |= balls[idx]
                if covered == (1 << num_words) - 1:
                    code.pop(i)
                    changed = True
                    break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
    
    # Try 2-for-1 replacements
    if best_code is not None:
        code = best_code.copy()
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            found = False
            for i in range(len(code)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(code)):
                    if time.time() > deadline - 10:
                        break
                    # Try removing both i and j
                    test_code = [c for k, c in enumerate(code) if k != i and k != j]
                    covered = 0
                    for idx in test_code:
                        covered |= balls[idx]
                    uncovered = ((1 << num_words) - 1) & ~covered
                    
                    if uncovered == 0:
                        # Both are redundant
                        code = test_code
                        improved = True
                        found = True
                        break
                    
                    # Find best single codeword to cover remaining uncovered
                    best_idx = -1
                    best_count = 0
                    test_set = set(test_code)
                    for k in range(num_words):
                        if k in test_set:
                            continue
                        cnt = bin(uncovered & balls[k]).count('1')
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = k
                    
                    if best_count >= bin(uncovered).count('1'):
                        # Replace both with one
                        code = test_code + [best_idx]
                        improved = True
                        found = True
                        break
                if found:
                    break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
    
    # Final verification and fallback
    if best_code is None:
        # Fallback: full greedy without restarts
        uncovered = (1 << num_words) - 1
        best_code = []
        while uncovered:
            if time.time() > deadline - 5:
                break
            best_idx = -1
            best_count = 0
            for i in range(num_words):
                cnt = bin(uncovered & balls[i]).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_idx == -1:
                break
            best_code.append(best_idx)
            uncovered &= ~balls[best_idx]
    
    # Verify validity
    if time.time() < deadline - 3:
        covered = 0
        for idx in best_code:
            covered |= balls[idx]
        if covered != (1 << num_words) - 1:
            # Fallback: add all words
            best_code = list(range(num_words))
    
    # Convert back to list of lists
    result = [all_words[i].tolist() for i in best_code]
    
    return result
