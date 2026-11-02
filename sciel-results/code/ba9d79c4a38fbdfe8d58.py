import itertools
import random
import time

def construct(q, n, radius):
    # Assume q=3, n=6, radius=1 based on task description
    # But handle generally if possible.
    
    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    
    # Fallback generic greedy
    return _construct_generic(q, n, radius)

def _construct_ternary_6_1():
    # Generate initial code: Direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(h + list(tail))
            
    def to_int(word):
        val = 0
        for digit in word:
            val = val * 3 + digit
        return val

    def to_word(val):
        word = []
        for _ in range(6):
            word.append(val % 3)
            val //= 3
        return word[::-1]

    code_set = set(to_int(w) for w in initial_code)
    
    all_words = list(range(729))
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        word = to_word(w)
        nbs = [w]
        for i in range(6):
            for new_val in range(3):
                if new_val == word[i]:
                    continue
                new_word = word.copy()
                new_word[i] = new_val
                nbs.append(to_int(new_word))
        neighbors[w] = nbs

    coverage = [0] * 729
    
    def add_codeword(c):
        for nb in neighbors[c]:
            coverage[nb] += 1
            
    def remove_codeword(c):
        for nb in neighbors[c]:
            coverage[nb] -= 1

    for c in code_set:
        add_codeword(c)

    current_code = list(code_set)
    
    start_time = time.time()
    deadline = 145.0
    
    best_size = len(current_code)
    best_code = set(current_code)
    
    while time.time() < deadline:
        # Calculate importance: min coverage in neighborhood
        # Higher min coverage = safer to remove
        importances = []
        for c in current_code:
            min_cov = min(coverage[nb] for nb in neighbors[c])
            importances.append((min_cov, c))
        
        # Sort by importance descending
        importances.sort(key=lambda x: x[0], reverse=True)
        
        removed_any = False
        for min_cov, c in importances:
            if time.time() > deadline:
                break
            # Re-check min_cov? It might have changed if previous removals happened in this pass?
            # We are iterating over a snapshot of importances.
            # If we remove a codeword, coverage changes.
            # The `min_cov` for remaining codewords might have decreased.
            # So we should re-evaluate or just try and check validity.
            
            remove_codeword(c)
            if min(coverage) > 0:
                current_code.remove(c)
                removed_any = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)
            else:
                add_codeword(c)
                
        if not removed_any:
            # Stuck. Try random swaps.
            if current_code:
                # Pick a few random codewords to try swapping
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    c_to_swap = random.choice(current_code)
                    new_c = random.randint(0, 728)
                    if new_c in current_code:
                        continue
                    
                    remove_codeword(c_to_swap)
                    add_codeword(new_c)
                    
                    if min(coverage) > 0:
                        current_code.remove(c_to_swap)
                        current_code.append(new_c)
                        # Don't update best_code here unless size decreased? 
                        # Size is same. But structure changed.
                        # We will try to remove in next loop iteration.
                        break
                    else:
                        remove_codeword(new_c)
                        add_codeword(c_to_swap)

    return [to_word(c) for c in best_code]

def _construct_generic(q, n, radius):
    # Simple greedy for fallback
    # This is not optimized for the specific task but ensures correctness for other params
    total = q ** n
    words = [list(map(int, itertools.product(range(q), repeat=n))[i]) for i in range(total)]
    
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    covered = [False] * total
    code = []
    
    # Precompute coverage sets for all words? Too big if n is large.
    # For small n, it's fine.
    # But this is a fallback, so keep it simple.
    
    remaining = set(range(total))
    
    while remaining:
        # Find word that covers most remaining
        best_w = None
        best_count = -1
        for w_idx in remaining:
            w = words[w_idx]
            count = 0
            for r_idx in remaining:
                if dist(w, words[r_idx]) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_w = w_idx
                
        code.append(words[best_w])
        # Update remaining
        w = words[best_w]
        new_remaining = set()
        for r_idx in remaining:
            if dist(w, words[r_idx]) > radius:
                new_remaining.add(r_idx)
        remaining = new_remaining
        
    return code
