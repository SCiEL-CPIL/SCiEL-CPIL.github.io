import time
import itertools

def construct(q, n, radius):
    # Parameters
    # q = 3, n = 8, radius = 1
    
    # 1. Generate initial code using direct sum of tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all tails of length n-4
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    # Construct initial codewords
    codewords = []
    for h in tetracode:
        for tail in tails:
            codewords.append(h + list(tail))
            
    # Map codeword to integer index in [0, q**n)
    def word_to_int(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx
        
    def int_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
            
    # Precompute neighbors for all possible words? 
    # Space size is 3^8 = 6561.
    # Neighbors of a word w: w itself and words differing in 1 position.
    # We can generate neighbors on the fly or precompute.
    # Precomputing neighbors for all 6561 words is feasible.
    
    all_words = [int_to_word(i) for i in range(q**n)]
    neighbors = {}
    for w in all_words:
        w_int = word_to_int(w)
        neigh = [w_int]
        for i in range(n):
            orig_val = w[i]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_copy = w[:]
                w_copy[i] = new_val
                neigh.append(word_to_int(w_copy))
        neighbors[w_int] = neigh
        
    # Initialize coverage counts
    coverage = [0] * (q**n)
    current_codewords = codewords[:]
    
    # Update coverage for initial code
    for c in current_codewords:
        c_int = word_to_int(c)
        for nb in neighbors[c_int]:
            coverage[nb] += 1
            
    # Redundancy removal loop
    # We want to remove codewords that are redundant.
    # A codeword c is redundant if for all nb in neighbors[c_int], coverage[nb] > 1.
    
    # To handle the order dependence, we can do multiple passes.
    # Also, we can try to remove codewords in a specific order.
    # Let's just iterate until no more removals.
    
    changed = True
    while changed:
        changed = False
        # Iterate in reverse to handle index shifting if we remove
        # Or collect indices to remove
        to_remove = []
        for i in range(len(current_codewords)):
            c = current_codewords[i]
            c_int = word_to_int(c)
            is_redundant = True
            for nb in neighbors[c_int]:
                if coverage[nb] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                to_remove.append(i)
                
        if to_remove:
            # Remove from largest index to smallest to keep indices valid
            # Or just filter
            new_codewords = []
            remove_set = set(to_remove)
            for i in range(len(current_codewords)):
                if i in remove_set:
                    # Update coverage
                    c = current_codewords[i]
                    c_int = word_to_int(c)
                    for nb in neighbors[c_int]:
                        coverage[nb] -= 1
                else:
                    new_codewords.append(current_codewords[i])
            current_codewords = new_codewords
            changed = True
            
    return current_codewords
