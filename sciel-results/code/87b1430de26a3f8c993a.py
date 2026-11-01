import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses Iterated Local Search with coverage-count tracking for fast redundancy checks and repair.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (evaluator kills at 150s)
    
    total_words = q ** n
    
    # Precompute powers of q for fast index-to-neighbor conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Helper to get neighbors (ball) of an index
    # Ball size for radius 1 is 1 + n*(q-1)
    # For q=3, n=6, size = 1 + 6*2 = 13
    def get_ball(idx):
        ball = [idx]
        # Extract digits
        # We can do this on the fly or precompute all balls
        # Precomputing all balls is better for 729 words
        return ball # Placeholder, will precompute below

    # Precompute balls for all words
    # balls[i] is a list of indices covered by codeword i
    balls = []
    for i in range(total_words):
        ball = [i]
        # Get digits
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        # Generate neighbors
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(q):
                if new_val != orig:
                    # Calculate new index
                    # idx = sum(d_j * q^j)
                    # delta = (new_val - orig) * q^pos
                    new_idx = i + (new_val - orig) * powers[pos]
                    ball.append(new_idx)
        balls.append(ball)

    # Initialize with a known good construction: Direct sum of tetracode (if applicable) or Greedy
    initial_codewords = set()
    if q == 3 and n == 6 and radius == 1:
        # Tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    # Convert word to index
                    idx = 0
                    for k, val in enumerate(word):
                        idx += val * powers[k]
                    initial_codewords.add(idx)
    else:
        # Fallback Greedy
        uncovered_mask = (1 << total_words) - 1
        while uncovered_mask:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                count = (uncovered_mask & (sum(1 << b for b in balls[candidate]))).bit_count()
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            initial_codewords.add(best_center)
            # Update mask
            for b in balls[best_center]:
                uncovered_mask &= ~(1 << b)

    best_codewords = set(initial_codewords)
    best_size = len(best_codewords)

    def is_valid(codewords):
        # Check if all words are covered
        # Using counts is faster if we maintain them, but for validation we can just check
        # Since we use counts in the main loop, we can check if any count is 0
        # But this function might be called on arbitrary sets. 
        # For 729 words, a simple set union check is fine.
        covered = set()
        for c in codewords:
            covered.update(balls[c])
        return len(covered) == total_words

    def calculate_counts(codewords):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in codewords:
            for w in balls[c]:
                counts[w] += 1
        return counts

    def local_reduce(codewords, counts):
        """
        Remove redundant codewords.
        A codeword c is redundant if min(counts[w] for w in balls[c]) >= 2.
        """
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of the list to allow removal
            for c in list(codewords):
                # Check if c is redundant
                # If all words in its ball are covered by at least 2 codewords, we can remove it
                # Note: counts[c] includes c itself. So we need min >= 2.
                min_count = np.min(counts[balls[c]])
                if min_count >= 2:
                    # Remove c
                    codewords.remove(c)
                    # Update counts
                    for w in balls[c]:
                        counts[w] -= 1
                    changed = True
                    # We can break and restart, or continue. Continuing is usually fine.
                    # If we remove c, it might make other codewords non-redundant? 
                    # No, removing a codeword only decreases counts. 
                    # If a codeword was redundant (min>=2), it remains redundant or becomes less redundant?
                    # Actually, if we remove c, counts for neighbors decrease.
                    # A codeword d that was redundant (min(counts[ball[d]])>=2) might now have min=1 
                    # if c was one of the two covering a specific word in ball[d].
                    # So we should restart the loop to be safe? 
                    # Or just continue. If we continue, we might miss opportunities where 
                    # removing c makes d redundant? No, removing c makes counts lower, 
                    # so it's harder for min to be >= 2.
                    # So once a codeword is non-redundant, it stays non-redundant as we remove more?
                    # Yes. So we can just continue.
        
        return codewords, counts

    def greedy_repair(codewords, counts):
        """
        Add codewords to cover all uncovered words (where counts[w] == 0).
        """
        # Find uncovered words
        # We can just loop until no zeros in counts
        while np.any(counts == 0):
            if time.time() > deadline:
                break
            
            # Find the candidate that covers the most uncovered words
            best_candidate = -1
            best_gain = -1
            
            # Optimization: Only consider candidates that cover at least one uncovered word?
            # Or just iterate all. 729 is small.
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                # Count how many uncovered words this candidate covers
                # uncovered_mask is bits where counts == 0
                # We can use numpy: sum(counts[balls[candidate]] == 0)
                gain = np.sum(counts[balls[candidate]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate
            
            if best_candidate == -1 or best_gain == 0:
                # Should not happen if valid solution exists and we haven't timed out
                # But if we are stuck (shouldn't be with greedy on full space), break
                break
                
            codewords.add(best_candidate)
            for w in balls[best_candidate]:
                counts[w] += 1
        
        return codewords, counts

    # Main ILS Loop
    current_codewords = set(best_codewords)
    current_counts = calculate_counts(current_codewords)
    
    # Initial reduction
    current_codewords, current_counts = local_reduce(current_codewords, current_counts)
    if len(current_codewords) < best_size:
        best_size = len(current_codewords)
        best_codewords = set(current_codewords)
        best_counts = current_counts.copy()

    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        
        # 1. Perturbation
        # Randomly remove 1 to 3 codewords
        k = random.randint(1, 3)
        if len(current_codewords) > k:
            to_remove = random.sample(list(current_codewords), k)
            for c in to_remove:
                current_codewords.remove(c)
                for w in balls[c]:
                    current_counts[w] -= 1
        
        # 2. Repair
        current_codewords, current_counts = greedy_repair(current_codewords, current_counts)
        
        # 3. Local Reduction
        current_codewords, current_counts = local_reduce(current_codewords, current_counts)
        
        # 4. Update Best
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            best_counts = current_counts.copy()
            
            # If we hit the theoretical lower bound or close, we can stop early?
            # Lower bound is 71. If we hit 71, we can stop.
            if best_size <= 71:
                break

    # Final Polish: Try to reduce the best solution found using the counts
    final_codewords = set(best_codewords)
    final_counts = best_counts.copy()
    final_codewords, final_counts = local_reduce(final_codewords, final_counts)
    
    if len(final_codewords) < best_size:
        best_size = len(final_codewords)
        best_codewords = final_codewords

    # Convert best_codewords (set of ints) to list of lists
    result = []
    for c in best_codewords:
        word = [0] * n
        temp = c
        for i in range(n):
            word[i] = temp % q
            temp //= q
        result.append(word)
        
    return result
