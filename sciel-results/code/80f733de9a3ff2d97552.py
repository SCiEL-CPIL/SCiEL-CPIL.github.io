import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for the 150s limit
    
    # 1. Initialize with a valid covering: all-zero word covers only itself and neighbors.
    # To be safe and fast, we start with a greedy set cover from scratch.
    # However, a pure greedy set cover on 2187 points might be slow if done naively.
    # Let's use a heuristic initialization: 
    # A simple valid code is the repetition code? No, that's not a covering.
    # Let's use the direct sum of a smaller perfect code if available, or just greedy.
    # For q=3, n=7, radius=1:
    # Total space = 3^7 = 2187.
    # A single codeword covers 1 + 7*(3-1) = 15 points.
    # Lower bound ~ 2187/15 ~ 146.
    
    # Strategy: 
    # 1. Generate a random set of codewords that is guaranteed to cover the space? 
    #    Actually, it's easier to start with a known valid code and reduce it.
    #    The seed program uses the tetracode product, which gives 729 words.
    #    We can start with that, or generate a random valid cover.
    #    A random valid cover is hard to generate directly.
    #    Let's start with the seed's 729-word code, as it's guaranteed valid.
    
    # Recreate the seed's code for initialization
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    init_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to numpy array for fast operations
    # Code is a list of lists. Let's keep it as a list of tuples or numpy array.
    # Numpy array is faster for Hamming distance calculations.
    code = np.array(init_code, dtype=np.uint8)
    
    # Precompute coverage of each codeword? 
    # Space is 3^7 = 2187. 
    # We can represent the space as a bitset or a boolean array of size 2187.
    # Let's map each word in {0,1,2}^n to an index 0..2186.
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
        
    # Precompute neighbors for each possible word?
    # There are 2187 words. Each has 1 + 7*2 = 15 neighbors.
    # Total neighbor entries = 2187 * 15 = 32805. This is small.
    # Let's precompute the list of indices covered by each possible codeword.
    
    # Actually, we only need to know if a codeword is redundant.
    # A codeword c is redundant if every point in its coverage set is also covered by some other codeword in the set.
    
    # Efficient approach:
    # Maintain a count array `coverage_count` of size 3^n.
    # coverage_count[idx] = number of codewords in current code that cover idx.
    # A codeword c is redundant if for all x in N(c), coverage_count[x] > 1.
    # If we remove c, we decrement coverage_count[x] for all x in N(c).
    # If any coverage_count[x] becomes 0, the removal is invalid.
    
    # Precompute neighbors for each possible codeword value (0..2186)
    # Note: codewords are vectors. We need to map vector -> index.
    # And for each index, we need the list of indices it covers (itself + neighbors).
    
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    # all_words shape (2187, 7)
    
    # Compute index for each word in all_words
    # This is just 0..2186 if we use standard base-q expansion.
    # Let's verify:
    # word [0,0,...,0] -> 0
    # word [0,0,...,1] -> 1
    # ...
    # word [2,2,...,2] -> 3^7 - 1 = 2186.
    # Yes, the index is the base-q number.
    
    # Precompute coverage sets for each possible codeword index
    # cover_sets[i] is a list of indices covered by the word with index i.
    cover_sets = [None] * (q**n)
    
    for i in range(q**n):
        w = all_words[i]
        covered = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                # Create new word
                new_w = w.copy()
                new_w[pos] = val
                idx = 0
                for k in range(n):
                    idx = idx * q + new_w[k]
                covered.append(idx)
        cover_sets[i] = covered
        
    # Initialize coverage counts
    # We start with the init_code.
    coverage_count = np.zeros(q**n, dtype=np.int32)
    
    # We need to map the codewords in init_code to their indices.
    # And keep track of which codewords are active.
    
    active_codewords = []
    for cw in init_code:
        idx = 0
        for k in range(n):
            idx = idx * q + cw[k]
        active_codewords.append(idx)
        
    # Update coverage counts
    for idx in active_codewords:
        for nb in cover_sets[idx]:
            coverage_count[nb] += 1
            
    current_code_size = len(active_codewords)
    
    # Greedy removal loop
    # We try to remove codewords one by one.
    # To be efficient, we can prioritize codewords that cover the "most" unique points?
    # Or just random order.
    # Let's try random shuffling and iterating.
    
    # We need to be careful with time.
    # Each removal check is O(15) if we just check the counts.
    # But we need to verify if removal is safe.
    # Safe if min(coverage_count[nb] for nb in cover_sets[idx]) > 1.
    
    # Let's implement a local search:
    # 1. Try to remove any codeword that is redundant.
    # 2. If no removal is possible, try to swap? Or just stop.
    # Given the time limit, simple greedy removal is a good start.
    
    # To speed up, we can maintain a list of "candidates" for removal.
    # A codeword is a candidate if it might be redundant.
    # Initially, all are candidates.
    
    # We'll use a while loop with a time check.
    improved = True
    while improved:
        improved = False
        # Shuffle active codewords to avoid bias
        random.shuffle(active_codewords)
        
        i = 0
        while i < len(active_codewords):
            if time.time() > deadline:
                break
                
            idx = active_codewords[i]
            # Check if redundant
            # If all neighbors have coverage_count > 1, we can remove it.
            # Optimization: precompute min coverage for this idx? No, counts change.
            
            is_redundant = True
            for nb in cover_sets[idx]:
                if coverage_count[nb] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                for nb in cover_sets[idx]:
                    coverage_count[nb] -= 1
                active_codewords.pop(i)
                improved = True
                # Do not increment i, because the next element shifted into i
                continue
            else:
                i += 1
                
        if time.time() > deadline:
            break
            
    # Convert back to list of lists
    result = []
    for idx in active_codewords:
        # Convert index back to word
        w = [0] * n
        for k in range(n-1, -1, -1):
            w[k] = idx % q
            idx //= q
        result.append(w)
        
    return result
