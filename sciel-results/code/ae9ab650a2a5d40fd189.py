import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # This task is specifically for q=3, n=7, radius=1
    # But the interface is general. We should handle the general case or at least the specific one efficiently.
    # Given the context, it's highly likely to be tested on 3,7,1.
    # However, to be safe, if q, n, radius differ, we might need a fallback.
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # And the task is "Football pool problem for 7 matches: ternary covering code K_3(7,1)".
    # So q=3, n=7, radius=1 is the target.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _generic_greedy(q, n, radius)

    # Specific optimization for 3, 7, 1
    total_words = q ** n  # 2187
    
    # Map each word (tuple of ints) to an index 0..2186
    # Index is the base-3 number represented by the word.
    # word (w0, ..., w6) -> w0*3^6 + ... + w6*3^0
    
    # Precompute the list of words
    words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % 3)
            x //= 3
        words.append(tuple(reversed(w)))
    
    # Precompute neighborhood masks
    # mask[i] is an integer where bit j is 1 if word[j] is in the neighborhood of word[i]
    masks = [0] * total_words
    for i in range(total_words):
        w = words[i]
        mask = 0
        # Self
        mask |= (1 << i)
        # Neighbors at distance 1
        for pos in range(n):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                # Find index of new_w
                # Since we generated words in base-3 order, index is sum(new_w[k] * 3^(n-1-k))
                idx = 0
                for k in range(n):
                    idx = idx * 3 + new_w[k]
                mask |= (1 << idx)
        masks[i] = mask

    all_mask = (1 << total_words) - 1

    def greedy_random():
        uncovered = all_mask
        code = []
        # List of indices not yet chosen? Or just iterate all?
        # Iterating all is fine for 2187.
        while uncovered:
            best_gain = -1
            best_candidates = []
            for i in range(total_words):
                # If word i is already in code, its gain is 0 (or we can skip it)
                # Actually, adding the same word again is useless.
                # But we don't track "in code" easily in this loop.
                # We can just check gain. If gain is 0, it's not best unless all are 0 (impossible if uncovered != 0).
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            # Pick one randomly from best candidates to allow different paths
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered &= ~masks[chosen]
        
        return code

    def is_valid_code(code):
        covered = 0
        for i in code:
            covered |= masks[i]
        return covered == all_mask

    def reduce_code(code):
        # Try to remove redundant codewords
        changed = True
        while changed:
            changed = False
            for idx in range(len(code)):
                # Try removing code[idx]
                temp_code = code[:idx] + code[idx+1:]
                if is_valid_code(temp_code):
                    code = temp_code
                    changed = True
                    break # Restart the loop as indices shifted
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple randomized greedy instances
    while time.time() < deadline:
        code = greedy_random()
        code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # If we found the known best, we could stop early, but let's keep searching for better
        if best_size <= 186:
             # It's very hard to go below 186, but let's try a bit more if time permits
             # Actually, 186 is the best known. Let's just run a few more to be sure.
             pass

    # Convert indices back to list of lists
    if best_code is None:
        # Fallback
        best_code = greedy_random()
        
    result = []
    for idx in best_code:
        # Convert index to base-3 digits
        w = []
        x = idx
        for _ in range(n):
            w.append(x % 3)
            x //= 3
        w.reverse()
        result.append(w)
        
    return result

def _generic_greedy(q, n, radius):
    # Simple fallback for other parameters
    # This is not optimized and may be slow for large n
    from itertools import product
    
    total = q ** n
    if total > 10000:
        # Too large for simple bitset approach in Python without numpy optimization
        # Just return a trivial code (all words)
        return [list(p) for p in product(range(q), repeat=n)]
        
    words = [list(p) for p in product(range(q), repeat=n)]
    
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
    
    def neighborhood(w):
        # Returns list of indices within distance radius
        inds = set()
        for i, other in enumerate(words):
            if hamming_dist(w, other) <= radius:
                inds.add(i)
        return inds
    
    # Precompute neighborhoods
    neigh = [neighborhood(w) for w in words]
    
    uncovered = set(range(total))
    code = []
    
    while uncovered:
        best_gain = -1
        best_cand = -1
        for i, w in enumerate(words):
            gain = len(neigh[i] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_cand = i
        code.append(words[best_cand])
        uncovered -= neigh[best_cand]
        
    return code
