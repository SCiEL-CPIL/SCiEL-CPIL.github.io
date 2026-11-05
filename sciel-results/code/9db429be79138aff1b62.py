import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # The task is fixed: q=3, n=8, radius=1
    # But we handle it generally for small sizes if possible, or fallback to seed logic if too large.
    # For q=3, n=8, space is 6561.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to the seed logic for other parameters if they appear, 
        # though the task contract implies this specific instance.
        # We assume the evaluator passes the specific task parameters.
        # If parameters differ, we might not have a pre-computed bitset strategy.
        # For now, we focus on the specific case q=3, n=8, r=1.
        # If other cases are passed, we return the seed result to be safe.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    N = q ** n  # 6561
    word_size = 64
    num_words = (N + word_size - 1) // word_size
    
    # Precompute all balls (sets of words within radius 1 of a center)
    # Represent each ball as a bitset (array of uint64)
    balls = np.zeros((N, num_words), dtype=np.uint64)
    
    # Helper to convert word index to coordinates
    # We can iterate through all words to build balls
    # A word w has neighbors: w itself, and w +/- 1 in each coordinate (mod q)
    
    # To speed up, we can generate the bitsets.
    # Let's create a mapping from word index to its bit position.
    # bit_pos = index
    
    # Iterate through all potential centers
    for center_idx in range(N):
        # Convert center_idx to coordinates
        coords = [0] * n
        temp = center_idx
        for i in range(n - 1, -1, -1):
            coords[i] = temp % q
            temp //= q
            
        # Add center itself
        balls[center_idx, center_idx // word_size] |= (1 << (center_idx % word_size))
        
        # Add neighbors
        for i in range(n):
            for delta in [-1, 1]:
                new_coords = coords[:]
                new_coords[i] = (new_coords[i] + delta) % q
                # Convert back to index
                idx = 0
                for j in range(n):
                    idx = idx * q + new_coords[j]
                balls[center_idx, idx // word_size] |= (1 << (idx % word_size))

    # Initialize uncovered set to all 1s (all words are uncovered)
    uncovered = np.zeros(num_words, dtype=np.uint64)
    for i in range(num_words):
        uncovered[i] = np.uint64(0xFFFFFFFFFFFFFFFF)
    # Mask out the last word if N is not a multiple of 64
    if N % word_size != 0:
        # The last bitset contains bits for indices N to 64*num_words-1
        # We want to keep only the first N bits.
        # Actually, we just need to ensure bits >= N are 0.
        # Bits are 0..63 in the last word.
        # Valid bits are 0 .. (N % 64 - 1)
        valid_bits = N % word_size
        mask = (1 << valid_bits) - 1
        uncovered[-1] = np.uint64(mask)
        
    # If N is exactly multiple of 64, the last word is fully valid.
    if N % word_size == 0 and N > 0:
         # All bits in the last word are valid
         pass 

    solution = []
    
    # Greedy Set Cover
    while True:
        # Check time
        if time.time() > deadline:
            break
            
        # Check if uncovered is empty
        # Sum of bits in uncovered
        # We can just check if any element is non-zero
        if not np.any(uncovered):
            break
            
        # Find the ball with maximum intersection with uncovered
        # Intersection count for each ball:
        # We need to count bits in (balls[i] & uncovered)
        # This is expensive to do for all 6561 balls in pure python/numpy loops per iteration.
        # However, 6561 is small enough that we can vectorize the intersection and bit count.
        
        # Intersection: balls & uncovered (broadcasting uncovered to (N, num_words))
        # This creates a temporary array of size N x num_words, which is ~6561 * 103 ~ 676k uint64s. ~5MB.
        # This is acceptable.
        
        # Efficient bit counting:
        # We can sum the bits. But np.sum on uint64 might overflow or be slow.
        # Better: use a lookup table for 8-bit chunks or similar?
        # Or simply: np.count_nonzero doesn't work on bits.
        # We can use a trick:
        # x = balls & uncovered
        # We want popcount for each row.
        
        # Let's try a simpler approach first:
        # Since N is small, maybe we can just loop?
        # 6561 iterations * 103 ops = ~676k ops per step.
        # If we need ~500 steps, that's 300M ops. Might be too slow in Python.
        
        # Let's use NumPy vectorization for the intersection and bit count.
        # To count bits in uint64 array:
        # We can reshape to (-1, 8) and sum a lookup table, or use a specific function.
        # Actually, `np.sum` on boolean arrays works.
        # `x != 0` gives boolean array. Summing booleans counts non-zero elements.
        # But we need bit count, not element count.
        
        # Let's use a lookup table for 16-bit values.
        # Or, since we are using NumPy, we can do:
        # x = balls & uncovered
        # x = x.view(np.uint8).reshape(N, -1) # This is tricky with endianness.
        
        # Alternative:
        # We can precompute the popcount for all 256 bytes and use it.
        # But let's try the simplest vectorized bit count first.
        # There is no direct `np.bit_count` for arrays in older numpy versions, 
        # but we can do:
        # x = (balls & uncovered)
        # # Split into 8 bits
        # # This is getting complex.
        
        # Let's stick to a Python loop for the selection step but optimize the intersection check.
        # Actually, 6561 is small. Let's just loop.
        
        best_score = -1
        best_idx = -1
        
        # Optimization: We can precompute the popcount of the balls themselves? No, it changes with uncovered.
        
        # Let's try a hybrid:
        # 1. Compute intersection for all balls: inter = balls & uncovered
        # 2. We need popcount of each row.
        # Let's use a lookup table for bytes.
        
        # Precompute popcount for 0-255
        popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
        
        # inter = balls & uncovered
        # inter_bytes = inter.view(np.uint8).reshape(N, -1)
        # This view might have endianness issues or alignment issues.
        
        # Safer way:
        # We'll just loop. 6561 iterations is fast in C, but this is Python.
        # 6561 * 103 ~ 676k operations.
        # If we do this 500 times, 338M operations.
        # Python does ~10M simple ops/sec. So ~30 seconds.
        # This is tight but might work.
        
        # Let's optimize the loop:
        # Use `int.bit_count()` if available (Python 3.10+).
        # We can convert the numpy uint64 array to a list of Python ints?
        # That's slow.
        
        # Let's use the vectorized bit count via lookup table properly.
        # inter = balls & uncovered
        # We need to handle the uint64 -> bytes conversion carefully.
        # Assume little-endian.
        # inter_bytes = np.ascontiguousarray(inter).view(np.uint8).reshape(N, -1)
        # scores = popcount_table[inter_bytes].sum(axis=1)
        
        inter = balls & uncovered
        # Check if we can view as uint8
        try:
            inter_bytes = np.ascontiguousarray(inter).view(np.uint8).reshape(N, -1)
            scores = popcount_table[inter_bytes].sum(axis=1)
        except:
            # Fallback to loop
            scores = np.zeros(N, dtype=np.int32)
            for i in range(N):
                # Count bits in inter[i]
                # inter[i] is a numpy array of uint64
                # We can just sum the popcounts of each uint64
                # But we need a popcount for uint64.
                # Let's just use the table on the bytes of each word.
                # This fallback is likely not needed.
                pass

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        if best_score == 0:
            # No ball covers any new word? This shouldn't happen if uncovered is not empty.
            break
            
        # Add this ball to solution
        # We need the codeword corresponding to best_idx
        coords = [0] * n
        temp = best_idx
        for i in range(n - 1, -1, -1):
            coords[i] = temp % q
            temp //= q
        solution.append(coords)
        
        # Update uncovered
        uncovered = uncovered & ~balls[best_idx]
        
        # Check time again
        if time.time() > deadline:
            break

    # If we broke out due to time, we might have an incomplete solution.
    # The task requires a VALID solution (covers all words).
    # If we didn't finish, we should return the seed solution to ensure validity,
    # or check if our current solution is valid.
    # Given the greedy nature, it usually finishes quickly.
    # If it doesn't, the seed (729 words) is a valid fallback.
    
    # Check if solution is valid
    if solution:
        # Verify coverage
        # This is expensive. We assume greedy works.
        # If time ran out, we might return an invalid set.
        # To be safe, if time ran out, return seed.
        if time.time() > deadline:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             
        return solution
    else:
        # If no solution found (should not happen), return seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
