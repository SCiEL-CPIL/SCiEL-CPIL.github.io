import time
import heapq
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    total_words = q ** n
    
    # Precompute: for each potential codeword, which words does it cover?
    # A codeword c covers c itself and all words at Hamming distance <= radius
    # For radius=1, that's c and all words differing in exactly 1 position
    
    # To save memory/time, we can compute coverage on the fly or precompute
    # 2187 codewords * 15 covered words = ~32805 entries, very manageable
    
    # Represent words as integers 0..q^n-1 for efficiency
    def word_to_int(w):
        val = 0
        for i, d in enumerate(w):
            val = val * q + d
        return val
    
    def int_to_word(v):
        w = []
        for i in range(n):
            w.append(v % q)
            v //= q
        return w[::-1]
    
    # Precompute neighbors for each word (words at distance 1)
    # For each word and each position, the 2 alternative values
    all_words = list(range(total_words))
    
    # Precompute coverage: cov[c] = set of words covered by codeword c
    # But sets are expensive. Use a different approach.
    
    # Alternative: for greedy set cover, we need to know for each uncovered word,
    # which codewords cover it. Or for each codeword, how many uncovered words it covers.
    
    # Let's precompute for each codeword c, the list of words it covers (including itself)
    coverage = {}
    for c in range(total_words):
        cw = int_to_word(c)
        covered = [c]
        for pos in range(n):
            orig = cw[pos]
            for val in range(q):
                if val == orig:
                    continue
                nw = cw[:]
                nw[pos] = val
                covered.append(word_to_int(nw))
        coverage[c] = covered
    
    # Greedy set cover
    uncovered = set(all_words)
    codewords = []
    
    # For efficiency, maintain count of uncovered words per codeword
    # But updating this after each selection is expensive if done naively.
    # Instead, just recompute: for each remaining codeword, count uncovered words it covers.
    # With 2187 codewords and 15 covered words each, this is 2187*15 = 32805 checks per iteration.
    # We might need ~150-200 iterations, so ~6.5M checks total. Fine.
    
    while uncovered:
        if time.time() > deadline - 10:
            break
        
        best_c = None
        best_count = 0
        for c in range(total_words):
            if c in [word_to_int(w) for w in codewords]:  # This is slow, use a set
                continue
            count = 0
            for w in coverage[c]:
                if w in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c is None or best_count == 0:
            break
        
        codewords.append(int_to_word(best_c))
        # Remove covered words from uncovered
        for w in coverage[best_c]:
            uncovered.discard(w)
    
    # Convert to proper format
    result = [list(w) for w in codewords]
    
    # Verify coverage
    if len(uncovered) == 0:
        return result
    
    # If not fully covered (shouldn't happen), add more codewords
    for w in list(uncovered):
        if time.time() > deadline - 5:
            break
        ww = int_to_word(w)
        result.append(ww)
        for cov_w in coverage[word_to_int(ww)]:
            uncovered.discard(cov_w)
    
    return result
