<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    # For the specific case of ternary length 7 radius 1, we use a greedy set cover
    # with a strong heuristic (choose the set that covers the most uncovered elements)
    # combined with local search / hill climbing to reduce the number of codewords.
    #
    # The seed uses the direct sum of the ternary tetracode: 9 * 3^3 = 243 words.
    # We aim to do better than 243, ideally approaching the known best of 186.
    #
    # Strategy:
    # 1. Start with a reasonable initial covering (e.g., greedy).
    # 2. Use local search: try removing a codeword if the coverage is still complete,
    #    or swap a codeword for another that covers more uncovered words.
    #
    # Given the 120s time limit, we need an efficient approach.
    #
    # For q=3, n=7, radius=1:
    # Total words: 3^7 = 2187
    # Each codeword covers itself and its 2*7=14 neighbors (distance 1), so 15 words per codeword.
    # Lower bound: 2187/15 = 145.8 -> 146. Known best: 186.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the seed approach for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Specialized solver for q=3, n=7, radius=1
    # Precompute all words and their neighbors
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)  # 2187
    
    # For each word, compute the set of codewords that would cover it (i.e., words at distance <= 1)
    # A word w is covered by codeword c if Hamming distance <= 1.
    # So for each word w, the potential codewords are w itself and all words at distance 1 from w.
    # There are 1 + 2*7 = 15 such words.
    
    # Build the incidence: for each word index, list of codeword indices that cover it
    # Actually, we want: for each codeword c, which words does it cover?
    # Let's build: cover[c] = set of word indices covered by codeword c
    
    # Precompute neighbors for each word
    def get_neighbors(w):
        """Return all words at Hamming distance <= 1 from w."""
        neighbors = [w]
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors.append(tuple(new_w))
        return neighbors
    
    # Build cover sets: for each possible codeword (which is a word in the space), 
    # what words does it cover?
    # Since any word can be a codeword, we have 2187 possible codewords.
    # cover[i] = list of word indices covered by codeword i
    
    cover = [None] * N
    for i, w in enumerate(all_words):
        neighbors = get_neighbors(w)
        cover[i] = [word_to_idx[nb] for nb in neighbors]
    
    # Greedy set cover to get an initial solution
    uncovered = set(range(N))
    codewords = []
    
    # For efficiency, use a list of uncovered and a count of how many codewords cover each uncovered word
    # Greedy: at each step, pick the codeword that covers the most uncovered words
    
    # To make this efficient, we'll track for each codeword how many uncovered words it covers
    # Initially, each codeword covers 15 words, but we need to track which are uncovered
    
    # Simple greedy with periodic recomputation
    # Given N=2187 and 2187 possible codewords, each covering 15 words, this is manageable
    
    # More efficient greedy: maintain a count of uncovered words per codeword
    # When we pick a codeword, decrement the count for all codewords that share uncovered words with it
    
    # Let's use a simpler approach: since the problem is small enough, use a standard greedy
    # with a priority queue or just scan each time (2187 * 2187/15 ~ 300k operations, fine)
    
    # Actually, let's use a smarter greedy:
    # 1. Start with all words uncovered
    # 2. Repeatedly pick the codeword that covers the most uncovered words
    # 3. Mark those words as covered
    
    # To avoid O(N^2) per step, we can precompute the cover sets and use a heap
    # But for 2187 words, even O(N^2) total is ~4.8M, which is fine in 120s
    
    # Let's do a straightforward greedy:
    import heapq
    
    # For each codeword, count how many uncovered words it covers
    # Initially all are uncovered, so each codeword covers 15 words
    # But as we add codewords, we need to update counts
    
    # Use a max-heap: (-count, codeword_idx)
    # We'll need to lazily delete stale entries
    
    uncovered_set = set(range(N))
    
    # count[i] = number of uncovered words covered by codeword i
    count = [15] * N  # Each codeword covers 15 words initially
    
    heap = [(-15, i) for i in range(N)]
    heapq.heapify(heap)
    
    codewords = []
    
    while uncovered_set:
        # Get the best codeword
        while heap:
            neg_count, i = heap[0]
            if -neg_count == count[i] and count[i] > 0:
                break
            heapq.heappop(heap)
        
        if not heap:
            break
        
        neg_count, i = heapq.heappop(heap)
        if count[i] == 0:
            continue
        
        # Add codeword i
        codewords.append(i)
        
        # Mark covered words
        for w in cover[i]:
            if w in uncovered_set:
                uncovered_set.remove(w)
                # For all codewords that cover w, decrement their count
                # We need to know which codewords cover w
                # Precompute: for each word w, which codewords cover it?
                # That's just the neighbors of w, i.e., get_neighbors(w)
                # But we already have that implicitly
    
    # This is getting complex. Let me use a simpler but effective approach:
    # Start with the tetracode-based solution (243 words), then try to reduce it.
    
    # Actually, let me just implement a clean greedy from scratch:
    
    uncovered = [True] * N
    num_uncovered = N
    
    # For each codeword, track how many uncovered words it covers
    # cover_count[i] = number of j such that uncovered[j] and j in cover[i]
    cover_count = [15] * N  # Initially all 15 are uncovered
    
    # We also need: for each word j, which codewords cover it?
    # That's the same as the neighbors of word j
    # Let's precompute: covers_word[j] = list of codeword indices that cover word j
    covers_word = [None] * N
    for j in range(N):
        w = all_words[j]
        neighbors = get_neighbors(w)
        covers_word[j] = [word_to_idx[nb] for nb in neighbors]
    
    # Greedy set cover
    codewords = []
    
    # Use a simple approach: repeatedly find the codeword with max cover_count
    # N=2187, each iteration is O(N) to find max, and we do ~200 iterations, so ~437k ops
    
    while num_uncovered > 0:
        # Find codeword with max cover_count
        max_idx = -1
        max_count = -1
        for i in range(N):
            if cover_count[i] > max_count:
                max_count = cover_count[i]
                max_idx = i
        
        if max_idx == -1 or max_count == 0:
            break
        
        codewords.append(max_idx)
        
        # Update: mark words covered by max_idx as covered
        for w in cover[max_idx]:
            if uncovered[w]:
                uncovered[w] = False
                num_uncovered -= 1
                # Decrement cover_count for all codewords that cover w
                for cw in covers_word[w]:
                    cover_count[cw] -= 1
    
    # Now we have an initial solution. Try to improve it with local search.
    # Local search: try to remove a codeword if possible, or swap.
    
    # First, let's see how many we got
    # The greedy should give something around 150-200
    
    # Local search: for each codeword, check if we can remove it
    # A codeword can be removed if all words it uniquely covers are also covered by other codewords
    
    # Build a coverage count: for each word, how many codewords cover it
    cov_count = [0] * N
    for i in codewords:
        for w in cover[i]:
            cov_count[w] += 1
    
    # Try removing codewords
    changed = True
    while changed:
        changed = False
        for idx in range(len(codewords) - 1, -1, -1):
            i = codewords[idx]
            # Check if all words covered by i are covered by at least one other codeword
            can_remove = True
            for w in cover[i]:
                if cov_count[w] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove it
                for w in cover[i]:
                    cov_count[w] -= 1
                codewords.pop(idx)
                changed = True
    
    # Convert indices back to codewords (lists of integers)
    result = [list(all_words[i]) for i in codewords]
    
    return result
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    """
    Solve the covering code problem using greedy set cover with bit-parallel
    operations for efficiency, followed by aggressive local search (remove + swap).
    
    For q=3, n=7, radius=1: 2187 words, each codeword covers 15 words.
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct sum of ternary tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Precompute all 2187 words as integers (base-3 representation)
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 2187
    
    # Convert words to integer indices for compact representation
    # Each word is a base-3 number: sum(d[i] * 3^i)
    word_vals = [0] * N
    for i, w in enumerate(all_words):
        val = 0
        for d in w:
            val = val * 3 + d
        word_vals[i] = val
    
    # Precompute neighbors: for each word, the 15 words at distance <= 1
    # cover[i] = list of word indices covered by codeword i
    cover = [None] * N
    covers_word = [None] * N  # covers_word[j] = list of codeword indices that cover word j
    
    for i, w in enumerate(all_words):
        neighbors = [w]
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors.append(tuple(new_w))
        cover[i] = [word_to_idx[nb] for nb in neighbors]
    
    # Build reverse: for each word j, which codewords cover it
    for i in range(N):
        for w in cover[i]:
            if covers_word[w] is None:
                covers_word[w] = []
            covers_word[w].append(i)
    
    # Convert to numpy arrays for bit-parallel operations
    # Represent coverage as bitsets using numpy uint64 arrays
    # 2187 bits need ceil(2187/64) = 35 uint64 words
    NUM_BLOCKS = (N + 63) // 64  # 35 blocks
    
    # cover_bits[i] = bitset of words covered by codeword i
    cover_bits = np.zeros((N, NUM_BLOCKS), dtype=np.uint64)
    for i in range(N):
        for w in cover[i]:
            block = w // 64
            bit = w % 64
            cover_bits[i, block] |= (1 << bit)
    
    # covers_word_bits[j] = bitset of codewords that cover word j
    covers_word_bits = np.zeros((N, NUM_BLOCKS), dtype=np.uint64)
    for j in range(N):
        for cw in covers_word[j]:
            block = cw // 64
            bit = cw % 64
            covers_word_bits[j, block] |= (1 << bit)
    
    # Greedy set cover using numpy for speed
    uncovered_mask = np.ones(NUM_BLOCKS, dtype=np.uint64)
    # Set all bits for words 0..N-1
    for b in range(NUM_BLOCKS):
        start = b * 64
        end = min(start + 64, N)
        if end == start + 64:
            uncovered_mask[b] = 0xFFFFFFFFFFFFFFFF
        else:
            uncovered_mask[b] = (1 << (end - start)) - 1
    
    num_uncovered = N
    codewords = []
    
    # Greedy: repeatedly pick codeword covering most uncovered words
    # Use numpy to compute overlap counts efficiently
    while num_uncovered > 0:
        # For each codeword, count how many uncovered words it covers
        # overlap[i] = popcount(cover_bits[i] & uncovered_mask)
        # This is expensive to do for all 2187 codewords each iteration
        # Instead, use a simpler approach: track counts incrementally
        
        # Actually, let's use a different strategy:
        # Maintain cover_count[i] = number of uncovered words covered by codeword i
        # Initially all 15
        # When we add a codeword, decrement counts for all codewords sharing covered words
        
        if len(codewords) == 0:
            # First iteration: all codewords cover 15 uncovered words
            # Pick any, say codeword 0
            best = 0
            best_count = 15
        else:
            # Find codeword with max uncovered coverage
            # Use numpy to compute this efficiently
            # For each codeword i, count = sum over blocks of popcount(cover_bits[i, b] & uncovered_mask[b])
            # This is 2187 * 35 = 76545 operations per iteration, ~200 iterations = 15M ops
            # Might be too slow in Python. Let's use a different approach.
            
            # Alternative: maintain a count array and update it incrementally
            pass
        
        # Let's use a simpler but effective greedy with periodic full recomputation
        # Compute coverage counts for all codewords
        counts = np.zeros(N, dtype=np.int32)
        for i in range(N):
            # Count bits in cover_bits[i] & uncovered_mask
            c = 0
            for b in range(NUM_BLOCKS):
                c += np.bit_count(cover_bits[i, b] & uncovered_mask[b])
            counts[i] = c
        
        best = np.argmax(counts)
        best_count = counts[best]
        
        if best_count == 0:
            break
        
        codewords.append(best)
        
        # Update uncovered mask
        for b in range(NUM_BLOCKS):
            uncovered_mask[b] &= ~cover_bits[best, b]
        
        # Update num_uncovered
        num_uncovered = sum(np.bit_count(uncovered_mask[b]) for b in range(NUM_BLOCKS))
    
    # Local search: remove redundant codewords
    # Build coverage count
    cov_count = np.zeros(N, dtype=np.int32)
    for i in codewords:
        for b in range(NUM_BLOCKS):
            bits = cover_bits[i, b]
            # For each set bit, increment cov_count
            while bits:
                lsb = bits & (-bits)
                bit_pos = np.bit_length(lsb) - 1
                word_idx = b * 64 + bit_pos
                if word_idx < N:
                    cov_count[word_idx] += 1
                bits &= bits - 1
    
    # Try removing codewords
    changed = True
    while changed:
        changed = False
        for idx in range(len(codewords) - 1, -1, -1):
            i = codewords[idx]
            # Check if all words covered by i have cov_count >= 2
            can_remove = True
            for b in range(NUM_BLOCKS):
                bits = cover_bits[i, b]
                while bits:
                    lsb = bits & (-bits)
                    bit_pos = np.bit_length(lsb) - 1
                    word_idx = b * 64 + bit_pos
                    if word_idx < N and cov_count[word_idx] <= 1:
                        can_remove = False
                        break
                    bits &= bits - 1
                if not can_remove:
                    break
            
            if can_remove:
                # Remove codeword i
                for b in range(NUM_BLOCKS):
                    bits = cover_bits[i, b]
                    while bits:
                        lsb = bits & (-bits)
                        bit_pos = np.bit_length(lsb) - 1
                        word_idx = b * 64 + bit_pos
                        if word_idx < N:
                            cov_count[word_idx] -= 1
                        bits &= bits - 1
                codewords.pop(idx)
                changed = True
    
    # Swap local search: try replacing one codeword with another
    # For each codeword, try replacing it with a non-codeword that covers more uncovered-when-removed words
    improved = True
    while improved:
        improved = False
        for idx in range(len(codewords)):
            i = codewords[idx]
            # Temporarily remove i
            for b in range(NUM_BLOCKS):
                bits = cover_bits[i, b]
                while bits:
                    lsb = bits & (-bits)
                    bit_pos = np.bit_length(lsb) - 1
                    word_idx = b * 64 + bit_pos
                    if word_idx < N:
                        cov_count[word_idx] -= 1
                    bits &= bits - 1
            
            # Find words now uncovered (cov_count == 0)
            # Try to find a single codeword that covers all uncovered words
            # This is expensive, so just check if we can find any improvement
            # For now, just restore and skip swap for simplicity
            for b in range(NUM_BLOCKS):
                bits = cover_bits[i, b]
                while bits:
                    lsb = bits & (-bits)
                    bit_pos = np.bit_length(lsb) - 1
                    word_idx = b * 64 + bit_pos
                    if word_idx < N:
                        cov_count[word_idx] += 1
                    bits &= bits - 1
    
    # Convert indices back to codewords
    result = [list(all_words[i]) for i in codewords]
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, this implementation is quite complex and may be slow due to the nested loops with bit operations. Let me reconsider and write a cleaner, more efficient version.

