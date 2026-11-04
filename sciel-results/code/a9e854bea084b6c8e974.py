import time
import random
import itertools
import sys

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    return _fallback_construct(q, n, radius)


def _fallback_construct(q, n, radius):
    # Simple fallback for non-standard cases
    if radius == 0:
        return [[i] * n for i in range(q)]
    # Greedy set cover for small instances
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    if radius != 1:
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        neighbors.append(nbrs)
    
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(total):
            if all(covered[j] for j in neighbors[i]):
                continue
            cnt = sum(1 for j in neighbors[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in neighbors[best_idx]:
            covered[j] = True
    
    return codewords


def _construct_ternary_8_1():
    q = 3
    n = 8
    total = q ** n  # 6561
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute neighbors using numpy-style arrays for speed
    # Each word is an index 0..6560, convert to base-3 digits
    def idx_to_digits(idx):
        digits = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx
    
    # Build neighbor list
    neighbors = [None] * total
    for idx in range(total):
        digits = idx_to_digits(idx)
        nbrs = [idx]
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val != orig:
                    digits[pos] = val
                    nbrs.append(digits_to_idx(digits))
            digits[pos] = orig
        neighbors[idx] = nbrs
    
    # Convert to sets for fast membership
    neighbor_sets = [set(nb) for nb in neighbors]
    
    # Build a good initial code using a known construction
    # Use the product of tetracode (length 4) with a covering code of length 4
    # Tetracode: { (a, b, a+b, a+2b) mod 3 : a,b in {0,1,2} }
    tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
    
    # For the second half, use all 9 words of a (4,1) ternary code
    # Actually, let's use a better initial: start with 486+ words from a known construction
    # Use the construction from van Laarhoven et al. if we can encode it
    
    # Alternative: use a greedy construction with smart ordering
    # Start with all 6561 words, greedily remove
    
    # Better: use a structured initial code
    # Code 1: tetracode × all 3^4 tails = 9 * 81 = 729 (same as seed)
    # Code 2: Use a different product structure
    
    # Let's try: partition into 9 groups of 729 words each
    # For each of 9 positions in first 4 coords, pick a covering of the last 4 coords
    
    # Actually, let's use a much smarter approach:
    # Start with a random set of ~500 words, then optimize
    
    best_code = None
    best_size = total
    
    for seed in range(50):
        if time.time() > deadline - 20:
            break
        
        random.seed(seed)
        
        # Generate initial code: random set of ~520 words
        initial_size = 520
        initial_code = random.sample(range(total), initial_size)
        code_set = set(initial_code)
        cover_count = [0] * total
        
        for c in initial_code:
            for nbr in neighbors[c]:
                cover_count[nbr] += 1
        
        # Check if valid
        valid = all(cover_count[i] >= 1 for i in range(total))
        
        if not valid:
            # Add random words until valid
            attempts = 0
            while not valid and attempts < 10000:
                c = random.randint(0, total - 1)
                if c not in code_set:
                    code_set.add(c)
                    for nbr in neighbors[c]:
                        cover_count[nbr] += 1
                    valid = all(cover_count[i] >= 1 for i in range(total))
                attempts += 1
            if not valid:
                continue
        
        code_list = list(code_set)
        cur_size = len(code_list)
        
        if cur_size < best_size:
            best_size = cur_size
            best_code = list(code_list)
        
        # Local search: try to remove words
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            random.shuffle(code_list)
            
            for i in range(len(code_list) - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                c = code_list[i]
                removable = True
                for nbr in neighbors[c]:
                    if cover_count[nbr] <= 1:
                        removable = False
                        break
                if removable:
                    code_list.pop(i)
                    for nbr in neighbors[c]:
                        cover_count[nbr] -= 1
                    improved = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
        
        # Try swaps to escape local minima
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            # Temporarily remove
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            
            # Find uncovered
            uncovered = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
            
            if not uncovered:
                code_list.pop(idx_c)
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                continue
            
            # Find replacement
            found = False
            # Try words that are neighbors of uncovered words
            candidates = set()
            for u in uncovered:
                for nb in neighbors[u]:
                    candidates.add(nb)
            
            for cp in candidates:
                if cp == c or cp in code_set:
                    continue
                # Check if cp covers all uncovered
                cp_nbrs = neighbor_sets[cp]
                all_cov = all(u in cp_nbrs for u in uncovered)
                if all_cov:
                    # Perform swap
                    code_set.discard(c)
                    code_set.add(cp)
                    code_list[idx_c] = cp
                    for nbr in neighbors[cp]:
                        cover_count[nbr] += 1
                    found = True
                    break
            
            if not found:
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
    
    if best_code is not None:
        return [idx_to_digits(c) for c in best_code]
    
    # Fallback: return a valid code
    return [idx_to_digits(i) for i in range(total)]
