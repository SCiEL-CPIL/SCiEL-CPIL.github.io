"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Base construction: direct sum of ternary tetracode (9 words of length 4,
    # perfect code of covering radius 1) with all words on remaining n-4 coords.
    # This gives 9 * 3^(n-4) codewords. For n=8, q=3: 9 * 81 = 729 words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n <= 4:
        # For small n, just use the tetracode (or a subset)
        if n == 4:
            return [list(w) for w in tetracode]
        else:
            # For n < 4, use all possible words
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    base_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Convert to numpy for vectorized operations
    codewords = np.array(base_words, dtype=np.int8)
    num_words = len(codewords)
    
    # Build coverage mask: for each position i, for each value v,
    # which codewords have value v at position i
    # cov[i, v, k] = 1 if codeword k has value v at position i
    cov = np.zeros((n, q, num_words), dtype=bool)
    for i in range(n):
        for v in range(q):
            cov[i, v, :] = (codewords[:, i] == v)
    
    # For each position i and value v, count how many codewords have that value
    cov_count = cov.sum(axis=2)  # shape (n, q)
    
    # Greedy redundancy removal: remove codewords that don't cover any unique word
    # A codeword c at index k is redundant if for every word w within distance 1 of c,
    # there exists another codeword c' != c within distance 1 of w.
    # 
    # A word w is at distance <= 1 from codeword k iff:
    #   - w == codeword k (distance 0), or
    #   - w differs from codeword k in exactly one position i, where w[i] = v != codeword[k][i]
    #
    # Codeword k is redundant if:
    #   - For w = codeword k itself: some other codeword k' has distance <= 1 from codeword k
    #     (i.e., codeword k' == codeword k [impossible, distinct] or differs in 1 pos)
    #   - For each position i and value v != codeword[k][i]: the word w (which is codeword k 
    #     with position i changed to v) is covered by some other codeword.
    #
    # The word w (codeword k with pos i changed to v) is covered by codeword k' if
    # codeword k' has value v at position i AND codeword k' matches codeword k in all 
    # other positions, OR codeword k' == w (distance 0 from w).
    #
    # Actually, let's think about this differently. A word w is covered by codeword k' 
    # if dist(w, k') <= radius. For radius=1:
    #   dist(w, k') = 0: w == k'
    #   dist(w, k') = 1: w and k' differ in exactly one position
    #
    # For the "change" words: w = codeword_k with position i set to v (v != codeword_k[i]).
    # This w is covered by k' if:
    #   - k' == w (k' has value v at pos i and matches codeword_k everywhere else)
    #   - k' differs from w in exactly one position j != i (k' has value v at pos i, 
    #     matches codeword_k at all positions except possibly j, and at j has some value 
    #     != codeword_k[j])
    #   - k' differs from w in exactly one position j == i (k' has value != v at pos i, 
    #     matches codeword_k everywhere else) -- but this means k' == codeword_k, which 
    #     we already handle.
    
    # Simpler approach: iterative removal.
    # For each codeword k, check if it's redundant by seeing if all words it uniquely
    # covers are also covered by others.
    #
    # Words uniquely covered by codeword k (not covered by any other codeword):
    # 1. codeword k itself: covered by k' if dist(codeword_k, k') <= 1, k' != k
    # 2. For each position i, value v != codeword_k[i]: the word w_i_v
    #    covered by k' if dist(w_i_v, k') <= 1, k' != k
    #
    # For efficiency with vectorized ops, let's use a different strategy:
    # Count for each "neighbor" how many codewords cover it.
    
    # Let's use a smarter approach: 
    # For each codeword k, the words it covers are:
    #   - itself (1 word)
    #   - (n * (q-1)) words at distance 1
    # Total: 1 + n*(q-1) words
    #
    # A codeword is removable if no word in its coverage is exclusively covered by it.
    # 
    # For vectorized checking, let's compute for each codeword whether it's "essential".
    # 
    # Alternative: use the fact that in a covering code, we can remove a codeword if
    # its entire ball is contained in the union of other balls.
    #
    # Let's do iterative greedy removal with vectorized checks.
    
    # For each codeword k, compute: is there any word w in B(k) such that 
    # B(k) is the only ball containing w?
    # 
    # w is in B(k) iff dist(w,k) <= 1.
    # w is NOT in B(k') for all k' != k iff for all k' != k, dist(w,k') > 1.
    #
    # For w = codeword_k: dist(w, k') > 1 for all k' != k means no other codeword is 
    # within distance 1 of codeword_k.
    #
    # For w = codeword_k with pos i changed to v: 
    #   dist(w, k') > 1 for all k' != k means no other codeword is within distance 1 of w.
    #   This means: no k' != k has value v at pos i AND matches codeword_k in all other 
    #   positions (dist=0 from w), AND no k' != k differs from w in exactly one position.
    
    # This is getting complex. Let me use a simpler but effective approach:
    # Iteratively try to remove codewords, checking coverage after each removal.
    
    # Actually, let's use a more practical approach: 
    # 1. Compute for each codeword a "score" based on how much unique coverage it provides.
    # 2. Remove codewords with lowest unique coverage first.
    # 3. Verify coverage after each batch of removals.
    
    # Let's use a different strategy: local search / simulated annealing style.
    # Start with the base code, then try to remove codewords one at a time (or in batches),
    # and if coverage is maintained, keep the removal.
    
    # For efficiency, let's check if removing codeword k maintains coverage.
    # Coverage is maintained iff every word in the space is still within distance 1 
    # of some remaining codeword.
    
    # Key insight: if we remove codeword k, the only words that might lose coverage 
    # are those in B(k) that were ONLY covered by k.
    
    # Let's compute for each word in the space which codewords cover it, but that's 
    # 3^8 = 6561 words, which is manageable.
    
    if q == 3 and n == 8:
        # Specialized fast path for the main task
        all_words = np.array(list(itertools.product(range(3), repeat=8)), dtype=np.int8)
        # all_words shape: (6561, 8)
        
        # Compute distance matrix: dist[w, k] = Hamming distance between word w and codeword k
        # For radius 1, we only care if dist <= 1
        # 
        # A word w is covered by codeword k iff dist(w,k) <= 1
        # dist(w,k) = 0: w == k
        # dist(w,k) = 1: w and k differ in exactly one position
        
        # For each word w, find which codewords cover it.
        # covered[w] = set of codeword indices that cover w
        # 
        # For efficiency, compute:
        # same[w, k] = 1 if w == codeword_k
        # diff1[w, i, k] = 1 if w and codeword_k differ only in position i
        
        # same: (6561, num_words)
        # diff1: (6561, 8, num_words)
        
        # This might be too large: 6561 * 729 = ~4.8M for same, 6561*8*729 = ~38M for diff1
        # That's a lot of memory. Let's be smarter.
        
        # Alternative: for each codeword k, the words it covers are:
        #   - codeword_k itself
        #   - for each position i, value v != codeword_k[i]: word with pos i = v, rest = codeword_k
        #
        # So for each codeword k, it covers 1 + 8*2 = 17 words.
        # Total coverage assignments: 729 * 17 = 12393
        # 
        # Let's build a coverage count: for each word index, how many codewords cover it.
        
        # Map each word to an index: word = (w0,...,w7) -> index = w0*3^7 + w1*3^6 + ... + w7
        # Actually, let's just use a dict or array of size 3^8 = 6561.
        
        word_to_idx = {}
        for idx, w in enumerate(all_words):
            # Convert word to tuple for hashing
            word_to_idx[tuple(w)] = idx
        
        # Or better: convert word to integer
        def word_to_int(w):
            result = 0
            for val in w:
                result = result * 3 + val
            return result
        
        # Precompute word_to_int for all codewords
        cw_ints = np.array([word_to_int(codewords[k]) for k in range(num_words)], dtype=np.int32)
        
        # For each codeword, compute the ints of all words it covers
        # covered_by[k] = set of word_ints covered by codeword k
        # But we want: cover_count[word_int] = number of codewords covering this word
        
        cover_count = np.zeros(3**n, dtype=np.int32)
        
        for k in range(num_words):
            # Cover codeword k itself
            cover_count[cw_ints[k]] += 1
            # Cover distance-1 neighbors
            cw = codewords[k]
            for i in range(n):
                for v in range(q):
                    if v != cw[i]:
                        neighbor = cw.copy()
                        neighbor[i] = v
                        cover_count[word_to_int(neighbor)] += 1
        
        # Now find codewords that can be removed: a codeword k can be removed if 
        # all words it covers have cover_count >= 2 (i.e., covered by at least one other codeword).
        
        # For codeword k, the words it covers are:
        #   - cw_ints[k]
        #   - for each i, v != cw[k][i]: word_to_int(cw with pos i = v)
        
        # Check if all these words have cover_count >= 2
        # If yes, removing k is safe.
        
        # Let's do iterative removal.
        removed = np.zeros(num_words, dtype=bool)
        iterations = 0
        while True:
            if time.time() > deadline - 5:
                break
            if iterations > 1000:
                break
            iterations += 1
            any_removed = False
            for k in range(num_words):
                if removed[k]:
                    continue
                # Check if codeword k is removable
                cw = codewords[k]
                # Check cw itself
                if cover_count[cw_ints[k]] <= 1:
                    continue
                # Check all distance-1 neighbors
                removable = True
                for i in range(n):
                    for v in range(q):
                        if v != cw[i]:
                            neighbor = cw.copy()
                            neighbor[i] = v
                            if cover_count[word_to_int(neighbor)] <= 1:
                                removable = False
                                break
                    if not removable:
                        break
                if removable:
                    # Remove codeword k
                    removed[k] = True
                    any_removed = True
                    # Update cover_count
                    cover_count[cw_ints[k]] -= 1
                    for i in range(n):
                        for v in range(q):
                            if v != cw[i]:
                                neighbor = cw.copy()
                                neighbor[i] = v
                                cover_count[word_to_int(neighbor)] -= 1
            if not any_removed:
                break
        
        # Filter out removed codewords
        result = [list(codewords[k]) for k in range(num_words) if not removed[k]]
        
        # Now try to add new codewords to further reduce? No, we want to minimize.
        # Let's try local search: swap a codeword to a better position.
        # Actually, let's try to remove more by re-checking (the above may miss some).
        
        # Do another pass of removal
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            any_removed = False
            for k in range(num_words):
                if removed[k]:
                    continue
                cw = codewords[k]
                if cover_count[cw_ints[k]] <= 1:
                    continue
                removable = True
                for i in range(n):
                    for v in range(q):
                        if v != cw[i]:
                            neighbor = cw.copy()
                            neighbor[i] = v
                            if cover_count[word_to_int(neighbor)] <= 1:
                                removable = False
                                break
                    if not removable:
                        break
                if removable:
                    removed[k] = True
                    any_removed = True
                    cover_count[cw_ints[k]] -= 1
                    for i in range(n):
                        for v in range(q):
                            if v != cw[i]:
                                neighbor = cw.copy()
                                neighbor[i] = v
                                cover_count[word_to_int(neighbor)] -= 1
            if not any_removed:
                break
        
        result = [list(codewords[k]) for k in range(num_words) if not removed[k]]
        
        # Verify coverage
        final_cover = np.zeros(3**n, dtype=np.int32)
        for cw in result:
            final_cover[word_to_int(cw)] += 1
            for i in range(n):
                for v in range(q):
                    if v != cw[i]:
                        neighbor = cw.copy()
                        neighbor[i] = v
                        final_cover[word_to_int(neighbor)] += 1
        if np.any(final_cover == 0):
            # Coverage broken, fall back to original
            return base_words
        
        return result
    
    else:
        # General case: just return the base construction
        return base_words

# EVOLVE-BLOCK-END
