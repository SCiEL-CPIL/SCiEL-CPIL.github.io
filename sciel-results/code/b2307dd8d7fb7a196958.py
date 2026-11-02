import time
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Precompute powers of 3
    pow3 = np.array([3**i for i in range(n)], dtype=np.int64)
    total = 3**n  # 6561
    
    # Generate all words as integers
    all_words = np.arange(total, dtype=np.int64)
    
    # Convert word integer to vector: w = sum c[i]*3^i
    # Precompute digit vectors for all words
    digits = np.zeros((total, n), dtype=np.int64)
    tmp = all_words.copy()
    for i in range(n):
        digits[:, i] = tmp % 3
        tmp //= 3
    
    # For each codeword, compute the set of covered words (itself + all radius-1 neighbors)
    # A radius-1 neighbor: change one coordinate to one of the other (q-1) values
    # For q=3: change one coordinate to the other 2 values
    
    # Precompute for each position i and each digit value d, the other values
    # covered_words[c] = sorted list of word integers covered by codeword c
    
    # Efficient: for each codeword c with digits d[0..n-1],
    # neighbors: for each position i, for each v in 0..q-1, v != d[i]:
    #   new_word = c + (v - d[i]) * 3^i
    
    # Build coverage matrix: for each codeword, which words does it cover?
    # covered_mask[c, w] = 1 if codeword c covers word w
    # But this is 6561 x 6561 = 43M entries, manageable as uint8
    
    # Actually, let's use a different approach:
    # For each word w, which codewords cover it?
    # A codeword c covers w if dist(c,w) <= 1
    # i.e., c = w or c differs from w in exactly one position
    
    # So for each word w, the set of codewords that could cover it is:
    # {w} union {w with one coordinate changed} = 1 + n*(q-1) = 17 codewords
    
    # Greedy set cover:
    # covered_by[w] = bitmask (or set) of codewords that have been selected and cover w
    # uncovered_count[c] = number of currently uncovered words that codeword c would cover
    
    # Let's use a numpy-based approach:
    # is_selected[c] = whether codeword c is in our solution
    # covered[w] = number of selected codewords covering w (or 0 if uncovered)
    
    # For speed, let's precompute: for each codeword c, the list of words it covers
    # cover_list[c] = array of word indices covered by codeword c (17 entries)
    
    cover_list = np.zeros((total, 1 + n*(q-1)), dtype=np.int64)
    for c in range(total):
        d = digits[c]
        idx = 0
        cover_list[c, idx] = c
        idx += 1
        for i in range(n):
            for v in range(q):
                if v != d[i]:
                    cover_list[c, idx] = c + (v - d[i]) * pow3[i]
                    idx += 1
    
    # Greedy set cover
    is_selected = np.zeros(total, dtype=bool)
    covered = np.zeros(total, dtype=np.int32)  # count of selected codewords covering each word
    
    # uncovered_count[c] = number of words w where covered[w]==0 and c covers w
    # Initialize: all words uncovered, so uncovered_count[c] = 17 for all c
    # But some words might be covered by multiple codewords... initially none selected
    
    # For efficiency, maintain uncovered_count as we go
    # When we select codeword c, for each w in cover_list[c]:
    #   if covered[w] was 0: uncovered_count decreases for all c' covering w
    #   covered[w] += 1
    
    # This is expensive. Alternative: just count uncovered words covered by each candidate.
    # For 6561 codewords, each covering 17 words, and 6561 total words:
    # Each iteration: compute for each c how many of its 17 words are uncovered
    # That's 6561 * 17 = 111K operations per iteration
    # We need ~486 iterations, so ~54M operations total. Should be fine.
    
    # Actually, let's be smarter. Use numpy.
    
    # uncovered[w] = True if word w is not yet covered
    uncovered = np.ones(total, dtype=bool)
    
    selected = []
    
    for step in range(total):
        if time.time() > deadline - 30:  # leave time for local search
            break
        
        # For each candidate codeword c, count how many uncovered words it covers
        # candidates_scores[c] = sum(uncovered[cover_list[c]])
        
        # cover_list is (total, 17), uncovered is (total,)
        # scores = cover_list -> index into uncovered -> sum
        scores = uncovered[cover_list].sum(axis=1)
        
        # Pick the codeword with maximum score
        best_c = np.argmax(scores)
        best_score = scores[best_c]
        
        if best_score == 0:
            break  # all covered
        
        # Select best_c
        is_selected[best_c] = True
        selected.append(best_c)
        
        # Update covered and uncovered
        for w in cover_list[best_c]:
            w = int(w)
            if uncovered[w]:
                covered[w] += 1
                uncovered[w] = False
            else:
                covered[w] += 1
    
    # Convert selected indices to codeword vectors
    if not selected:
        # Fallback: return all words
        return [digits[i].tolist() for i in range(total)]
    
    selected_codes = [digits[c].tolist() for c in selected]
    
    # Local search: try to remove redundant codewords
    # A codeword c is redundant if every word it covers is also covered by another selected codeword
    
    # Rebuild covered count from scratch
    covered_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        for w in cover_list[c]:
            covered_count[int(w)] += 1
    
    # Try removing each codeword
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            c = selected[i]
            # Check if removing c leaves all words it covers still covered
            # i.e., for all w in cover_list[c], covered_count[w] > 1
            cl = cover_list[c]
            if np.all(covered_count[cl] > 1):
                # Remove c
                for w in cl:
                    covered_count[int(w)] -= 1
                selected.pop(i)
                changed = True
    
    # Final local search: try swapping - replace one codeword with another to reduce count
    # This is more complex; skip for now
    
    selected_codes = [digits[c].tolist() for c in selected]
    
    return selected_codes
