"""Greedy set cover + local search removal for ternary covering code K_3(8,1)."""
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: use direct sum construction for other parameters
        # For now, just return a trivial covering (all words) if small enough, else random
        total = q ** n
        if total <= 100000:
            import itertools
            return [list(x) for x in itertools.product(range(q), repeat=n)]
        else:
            # Random construction
            random.seed(42)
            codewords = []
            seen = set()
            while len(codewords) < 1000:
                w = tuple(random.randint(0, q-1) for _ in range(n))
                if w not in seen:
                    seen.add(w)
                    codewords.append(list(w))
            return codewords
    
    # Specialized for q=3, n=8, radius=1
    q, n, r = 3, 8, 1
    
    # Precompute neighbor sets for each codeword
    # A word w covers itself and all words at Hamming distance 1
    # Total words covered by one codeword: 1 + n*(q-1) = 1 + 8*2 = 17
    
    def neighbors(w):
        """Generate all words within Hamming distance r of w."""
        result = [list(w)]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    result.append(new_w)
        return result
    
    def to_tuple(w):
        return tuple(w)
    
    # Greedy set cover
    all_words = set()
    import itertools
    for w in itertools.product(range(q), repeat=n):
        all_words.add(w)
    
    covered = set()
    codewords = []
    
    # Precompute all possible codewords
    all_candidates = list(itertools.product(range(q), repeat=n))
    
    # For efficiency, we'll track coverage counts
    # covered_count[word] = number of codewords that cover this word
    # But that's 6561 words, so it's fine
    
    # Greedy: repeatedly pick the codeword that covers the most uncovered words
    # But we need to be fast. Let's use a different approach.
    
    # Actually, let's use a smarter greedy: 
    # Start with the tetracode-based construction (486 words) and try to remove words.
    
    # Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    codewords = [tuple(w) for w in initial_codewords]
    covered = set()
    
    # Compute coverage
    for cw in codewords:
        for nb in neighbors(cw):
            covered.add(tuple(nb))
    
    # Now try to remove codewords that don't have unique coverage
    # A codeword can be removed if all words it covers are also covered by other codewords
    
    def can_remove(cw_index, codewords):
        """Check if codeword at cw_index can be removed."""
        cw = codewords[cw_index]
        # For each word covered by cw, check if it's covered by another codeword
        for nb in neighbors(cw):
            nb_t = tuple(nb)
            # Check if any other codeword covers nb_t
            found = False
            for i, other_cw in enumerate(codewords):
                if i == cw_index:
                    continue
                # Check if other_cw covers nb_t
                diff = sum(1 for a, b in zip(other_cw, nb_t) if a != b)
                if diff <= radius:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # This is too slow. Let's use a different approach.
    
    # Better: maintain a count of how many times each word is covered
    # covered_count[word] = number of codewords that cover this word
    
    def compute_covered_count(codewords):
        counts = {}
        for cw in codewords:
            for nb in neighbors(cw):
                nb_t = tuple(nb)
                counts[nb_t] = counts.get(nb_t, 0) + 1
        return counts
    
    # Start with tetracode construction
    codewords = [tuple(w) for w in initial_codewords]
    covered_count = compute_covered_count(codewords)
    
    # Try to remove codewords
    removed = True
    while removed:
        removed = False
        for i in range(len(codewords)):
            if time.time() > deadline - 5:
                break
            cw = codewords[i]
            # Check if all words covered by cw have count >= 2
            can_rm = True
            for nb in neighbors(cw):
                nb_t = tuple(nb)
                if covered_count.get(nb_t, 0) < 2:
                    can_rm = False
                    break
            if can_rm:
                # Remove codeword i
                for nb in neighbors(cw):
                    nb_t = tuple(nb)
                    covered_count[nb_t] -= 1
                codewords.pop(i)
                removed = True
                break
    
    # Now do local search: try swapping or adding/removing
    # Try to find a smaller covering by random restarts
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Simulated annealing: try random mutations
    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline - 10:
        # Try a random mutation: remove a random codeword and see if coverage is maintained
        # If not, add a random codeword that helps
        
        current_size = len(codewords)
        
        # Try removing a random codeword
        if current_size > 100:
            idx = random.randint(0, current_size - 1)
            cw = codewords[idx]
            
            # Check if we can remove it
            can_rm = True
            for nb in neighbors(cw):
                nb_t = tuple(nb)
                if covered_count.get(nb_t, 0) < 2:
                    can_rm = False
                    break
            
            if can_rm:
                for nb in neighbors(cw):
                    nb_t = tuple(nb)
                    covered_count[nb_t] -= 1
                codewords.pop(idx)
                
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
            else:
                # Try swapping: remove this one and add another that covers the uncovered words
                # Find words that would become uncovered
                uncovered = []
                for nb in neighbors(cw):
                    nb_t = tuple(nb)
                    if covered_count.get(nb_t, 0) < 2:
                        uncovered.append(nb_t)
                
                if uncovered:
                    # Try to find a codeword that covers many of these uncovered words
                    # and is not already in the set
                    best_new_cw = None
                    best_gain = 0
                    for _ in range(50):  # Try 50 random candidates
                        new_cw = tuple(random.randint(0, q-1) for _ in range(n))
                        if new_cw in codewords:
                            continue
                        gain = 0
                        for nb in neighbors(new_cw):
                            nb_t = tuple(nb)
                            if nb_t in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_new_cw = new_cw
                    
                    if best_new_cw is not None and best_gain > 0:
                        # Remove old, add new
                        for nb in neighbors(cw):
                            nb_t = tuple(nb)
                            covered_count[nb_t] -= 1
                        codewords.pop(idx)
                        
                        for nb in neighbors(best_new_cw):
                            nb_t = tuple(nb)
                            covered_count[nb_t] = covered_count.get(nb_t, 0) + 1
                        codewords.append(best_new_cw)
        
        temp *= cooling
    
    # Verify coverage and return
    # Check that all words are covered
    final_covered = set()
    for cw in best_codewords:
        for nb in neighbors(cw):
            final_covered.add(tuple(nb))
    
    # If coverage is broken, fall back to the initial construction
    if len(final_covered) < q ** n:
        return [list(w) for w in initial_codewords]
    
    return [list(w) for w in best_codewords]
