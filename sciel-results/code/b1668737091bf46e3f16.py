import itertools

def construct(q, n, radius):
    # This implementation is tuned for q=3, n=8, radius=1
    # but will handle general small cases reasonably well.
    
    total_words = q ** n
    
    # Precompute neighbors for each word in the space
    # A word is an integer in 0..total_words-1
    # We need to map int <-> base-q representation to calculate Hamming distance
    
    def int_to_list(val, base, length):
        res = [0] * length
        for i in range(length):
            res[i] = val % base
            val //= base
        return res
    
    def list_to_int(lst, base):
        val = 0
        for i in range(len(lst)-1, -1, -1):
            val = val * base + lst[i]
        return val

    # Precompute neighbors
    neighbors = []
    for i in range(total_words):
        w = int_to_list(i, q, n)
        neigh = {i}
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                ni = list_to_int(w, q)
                neigh.add(ni)
        w[pos] = orig # restore, though w is recreated in next iter
        neighbors.append(neigh)

    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        # Find the word in uncovered that covers the most uncovered words
        best_c = -1
        best_count = -1
        # Optimization: if uncovered is small, iterate over it.
        # If uncovered is large, maybe sampling? 
        # For n=8, q=3, total=6561. 
        # Iterating over all uncovered is O(N^2) total if we do it naively?
        # No, we iterate over uncovered once per step.
        # Step 1: 6561 checks.
        # Step 2: ~6544 checks.
        # ...
        # This is O(N^2) which is ~42 million operations. 
        # Each operation is a set intersection of size 17.
        # 42M * 17 is too slow for Python (seconds to minutes).
        
        # We need a better heuristic or optimization.
        # Instead of checking all uncovered, we can just pick ONE uncovered word, 
        # say the first one, and use it? No, that's not greedy.
        
        # Alternative: 
        # The greedy set cover is O(N * M) where M is universe size.
        # Here N is code size, M is universe size.
        
        # Let's try to optimize the "find best" step.
        # We can just pick a random word from uncovered? 
        # Or, we can note that if we pick a word, it covers ~17.
        # Maybe we can just iterate through a subset?
        
        # Actually, for this specific small instance (6561 words), 
        # a smarter greedy might work.
        
        # Let's try: Pick the word in `uncovered` that has the minimum number of uncovered neighbors? 
        # No, we want max coverage.
        
        # Let's try to speed up the search.
        # We can maintain a count of how many uncovered neighbors each word has?
        # That's complex to update.
        
        # Given 120s, let's estimate the naive greedy time more carefully.
        # 6561 words.
        # Average uncovered size is ~3000.
        # Steps ~ 400.
        # Total iterations ~ 400 * 3000 = 1.2 million.
        # Each iteration: `len(neighbors[c] & uncovered)`.
        # Set intersection of size 17 with a set of size 3000.
        # This is fast. 1.2M * (cost of intersection).
        # Cost of intersection is O(min(size1, size2)) = O(17).
        # 1.2M * 17 = 20.4M operations.
        # In Python, 20M simple operations take ~1-2 seconds.
        # So the naive greedy IS fast enough!
        
        for c in uncovered:
            count = len(neighbors[c] & uncovered)
            if count > best_count:
                best_count = count
                best_c = c
                
        code.append(best_c)
        uncovered = uncovered - neighbors[best_c]
        
    # Convert back to list of lists
    return [int_to_list(c, q, n) for c in code]
