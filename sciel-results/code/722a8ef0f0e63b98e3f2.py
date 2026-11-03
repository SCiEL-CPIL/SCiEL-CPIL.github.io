import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = q ** n
    # N = 6561 for q=3, n=8
    
    # Helper: convert index to word and back
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbors
    # For q=3, n=8, radius=1:
    # Each word has 1 + 8*2 = 17 neighbors.
    neighbors = np.zeros((N, 17), dtype=np.int32)
    # We need to fill this.
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)))
    
    # Map word to index
    # Since it's base q, index = sum(word[i] * q^(n-1-i))
    # We can compute indices directly from the product generation order if we match them.
    # itertools.product generates in lexicographic order, which matches base-q counting if we treat the first element as most significant.
    # Let's verify:
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> q
    # Yes.
    
    # So all_words[i] is the word for index i.
    
    # Compute neighbors
    for i in range(N):
        word = all_words[i]
        nbrs = [i]
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val != orig:
                    word[pos] = new_val
                    nbrs.append(word_to_idx(word))
            word[pos] = orig
        neighbors[i] = nbrs
        
    # Greedy Set Cover
    # uncovered: boolean array
    uncovered = np.ones(N, dtype=bool)
    # cover_count: how many times a word is covered by the current code
    cover_count = np.zeros(N, dtype=np.int32)
    
    code = []
    code_set = set()
    
    # To speed up greedy, we can maintain for each candidate c, the number of currently uncovered words it covers.
    # candidate_gain[c] = sum(1 for w in neighbors[c] if uncovered[w])
    # Initially, gain[c] = 17 for all c.
    
    while uncovered.any() and time.time() < deadline:
        # Find c that maximizes gain[c]
        # We can update gains incrementally, but for N=6561, a full scan is 6561 * 17 ~ 111k ops, which is fast.
        # Let's just compute the gain for the best candidate.
        
        best_c = -1
        best_gain = -1
        
        # Optimization: Only consider candidates that are not in code_set?
        # Actually, adding a codeword already in code_set is useless.
        # We can iterate through all c, but skip if c in code_set.
        
        # To make this faster, we can keep a list of "active" candidates?
        # Or just iterate all. 6561 iterations is trivial.
        
        for c in range(N):
            if c in code_set:
                continue
            # Calculate gain
            # gain = number of uncovered neighbors
            gain = np.sum(uncovered[neighbors[c]])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c == -1 or best_gain == 0:
            break
            
        code.append(best_c)
        code_set.add(best_c)
        
        # Update uncovered and cover_count
        for w in neighbors[best_c]:
            if uncovered[w]:
                uncovered[w] = False
            cover_count[w] += 1
            
    # Local Search: Reduction
    # Try to remove redundant codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            # Check if c is redundant
            # A word w is uncovered if cover_count[w] == 1 and w in neighbors[c]
            # So if no such w exists, c is redundant.
            is_redundant = True
            for w in neighbors[c]:
                if cover_count[w] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove c
                for w in neighbors[c]:
                    cover_count[w] -= 1
                code.pop(i)
                code_set.discard(c)
                improved = True
                break # Restart scan
    
    # Local Search: Swaps
    # Try to swap a codeword with another to potentially enable further reductions
    random.seed(42)
    max_swaps = 5000
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if len(code) < 2:
            break
            
        i = random.randint(0, len(code) - 1)
        old_c = code[i]
        
        # Pick a random new_c
        new_c = random.randint(0, N - 1)
        if new_c in code_set:
            continue
            
        # Check if swap is valid
        # Valid if all words uniquely covered by old_c are also covered by new_c
        # Unique covered by old_c: w in neighbors[old_c] and cover_count[w] == 1
        # These w must be in neighbors[new_c]
        
        valid = True
        for w in neighbors[old_c]:
            if cover_count[w] == 1:
                # w must be covered by new_c
                if w not in neighbors[new_c]:
                    valid = False
                    break
        
        if valid:
            # Perform swap
            for w in neighbors[old_c]:
                cover_count[w] -= 1
            for w in neighbors[new_c]:
                cover_count[w] += 1
            code[i] = new_c
            code_set.discard(old_c)
            code_set.add(new_c)
            
            # Try to reduce again
            for j in range(len(code)):
                if time.time() > deadline:
                    break
                cj = code[j]
                is_redundant = True
                for w in neighbors[cj]:
                    if cover_count[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in neighbors[cj]:
                        cover_count[w] -= 1
                    code.pop(j)
                    code_set.discard(cj)
                    break

    # Final verification
    covered = np.zeros(N, dtype=bool)
    for c in code:
        covered[neighbors[c]] = True
        
    if not covered.all():
        # Fallback: add missing
        uncovered = np.where(~covered)[0]
        while len(uncovered) > 0 and time.time() < deadline:
            # Greedy add
            best_c = -1
            best_gain = -1
            for c in range(N):
                if c in code_set:
                    continue
                gain = np.sum(uncovered[neighbors[c]])
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            code_set.add(best_c)
            for w in neighbors[best_c]:
                covered[w] = True
            uncovered = np.where(~covered)[0]

    return [idx_to_word(c) for c in code]
