import time
import itertools
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds limit to be safe (150s wall-clock kill)
    
    # Helper to convert word to int
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
    
    # Precompute all words as ints
    num_words = q ** n
    all_words_ints = list(range(num_words))
    
    # Precompute balls for all possible words
    # ball[word_int] = frozenset of word_ints within distance 1
    ball_cache = {}
    for w_int in all_words_ints:
        # Convert back to tuple
        w = []
        tmp = w_int
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        w = tuple(reversed(w))
        
        ball = {w_int}
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    new_w = list(w)
                    new_w[i] = val
                    ball.add(word_to_int(tuple(new_w)))
        ball_cache[w_int] = frozenset(ball)
    
    # 1. Generate seed codewords
    # Direct sum of tetracode (length 4) and all words of length n-4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    seed_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Convert to tuples for hashing, then to ints for efficiency
    codeword_ints = set(word_to_int(tuple(c)) for c in seed_codewords)
    
    # Maintain cover counts: for each word, how many codewords cover it
    cover_count = [0] * num_words
    
    # Initialize cover counts
    for c_int in codeword_ints:
        for w_int in ball_cache[c_int]:
            cover_count[w_int] += 1
    
    # Precompute int to tuple mapping for output
    def int_to_word(w_int):
        w = []
        tmp = w_int
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        return list(reversed(w))
    
    # Build inverted index: for each word, which codewords cover it
    word_to_codewords = [[] for _ in range(num_words)]
    for c_int in codeword_ints:
        for w_int in ball_cache[c_int]:
            word_to_codewords[w_int].append(c_int)
    
    # Greedy removal with priority
    # Priority: codewords with higher redundancy (more words covered multiple times) removed first
    
    def compute_redundancy(c_int):
        """Compute redundancy score for a codeword. Higher = more redundant = better to remove."""
        ball = ball_cache[c_int]
        redundancy = 0
        for w_int in ball:
            if cover_count[w_int] > 1:
                redundancy += cover_count[w_int] - 1
        return redundancy
    
    # Build initial priority queue
    # Use (priority, counter, codeword_int) to avoid comparing codeword_ints
    counter = 0
    pq = []
    for c_int in codeword_ints:
        ball = ball_cache[c_int]
        min_count = min(cover_count[w] for w in ball)
        if min_count > 1:
            redundancy = compute_redundancy(c_int)
            heapq.heappush(pq, (-redundancy, counter, c_int))
            counter += 1
    
    # Removal phase
    while pq and time.time() < deadline - 5:
        neg_red, _, c_int = heapq.heappop(pq)
        
        # Check if still in codewords
        if c_int not in codeword_ints:
            continue
        
        # Verify still removable
        ball = ball_cache[c_int]
        removable = True
        for w_int in ball:
            if cover_count[w_int] <= 1:
                removable = False
                break
        
        if not removable:
            continue
        
        # Remove this codeword
        codeword_ints.remove(c_int)
        
        # Update cover counts and inverted index
        for w_int in ball:
            cover_count[w_int] -= 1
            # Remove c_int from word_to_codewords[w_int]
            # This is O(len(list)) but lists are small
            if c_int in word_to_codewords[w_int]:
                word_to_codewords[w_int].remove(c_int)
        
        # Find affected codewords: those that share a word with c
        # These may have changed redundancy
        affected_codewords = set()
        for w_int in ball:
            for c_prime in word_to_codewords[w_int]:
                if c_prime != c_int:
                    affected_codewords.add(c_prime)
        
        # Add affected codewords to priority queue if they're removable
        for c_prime in affected_codewords:
            if c_prime not in codeword_ints:
                continue
            ball_prime = ball_cache[c_prime]
            min_count_prime = min(cover_count[w] for w in ball_prime)
            if min_count_prime > 1:
                redundancy_prime = compute_redundancy(c_prime)
                heapq.heappush(pq, (-redundancy_prime, counter, c_prime))
                counter += 1
    
    # Convert back to list of lists
    return [int_to_word(c_int) for c_int in codeword_ints]
