import time
import random

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses a specialized ILS with tetracode-based initializations and efficient local search.
    """
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors for all words
    # neighbors[i] is a list of indices covered by word i
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val

    # For efficiency, precompute a boolean matrix or use sets? 
    # Sets of neighbors are good for "is w in neighbors[c]" checks.
    # But for "how many uncovered does c cover", we need to iterate neighbors[c].
    # Let's keep neighbors as lists, but maybe create a set version for membership?
    # No, the lists are small (13 for q=3, n=6, r=1).
    
    # Tetracode construction for q=3, n=6, r=1
    # Tetracode is a (4, 9, 3) code? No, it's a covering code.
    # The tetracode consists of 9 words: (a, b, a+b, a+2b) for a,b in {0,1,2}.
    # It covers all 3^4 = 81 words with radius 1.
    # Direct sum with tail of length 2 gives 9 * 3^2 = 81 words.
    
    def generate_tetracode_init():
        """Generate the 81-word tetracode direct sum code."""
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = set()
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    code.add(idx)
        return code

    # Greedy initialization
    def greedy_init():
        uncovered = set(range(total_words))
        code = set()
        while uncovered:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                count = 0
                for nb in neighbors[candidate]:
                    if nb in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            code.add(best_center)
            for nb in neighbors[best_center]:
                if nb in uncovered:
                    uncovered.remove(nb)
        return code

    # Try a few random initializations or the tetracode one
    # For this specific task, tetracode is a strong start.
    # Let's also try a simple greedy one.
    
    initializations = []
    if q == 3 and n == 6 and radius == 1:
        initializations.append(generate_tetracode_init())
    # Always add greedy
    initializations.append(greedy_init())
    
    # Pick the best initial size
    best_initial = min(initializations, key=len)
    best_codewords = set(best_initial)
    best_size = len(best_codewords)
    
    # Local Search: Reduce
    def reduce_codewords(codewords):
        changed = True
        while changed:
            changed = False
            for c in list(codewords):
                is_redundant = True
                for w in neighbors[c]:
                    covered_by_other = False
                    for other_c in codewords:
                        if other_c == c:
                            continue
                        # Check if w is in neighbors[other_c]
                        # Optimization: Since neighbors are small, linear scan is fine.
                        if w in neighbors[other_c]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                if is_redundant:
                    codewords.remove(c)
                    changed = True
                    break
        return codewords

    # Local Search: Swap (Remove one, add one that covers more)
    def try_swap(codewords):
        # Try to replace one codeword with another to reduce size or keep size but improve?
        # Actually, we want to minimize size.
        # Strategy: Remove c, see how many are uncovered. Add best d to cover them.
        # If len(codewords) remains same or decreases?
        # If we remove c and add d, size is same.
        # This is useful to escape local minima where no single removal works.
        
        # Let's implement a 1-swap: remove c, add d.
        # To be an improvement, we don't just want to maintain coverage, we want to
        # enable a future reduction.
        # Simpler: Just do the ILS loop: Reduce, Kick, Repair.
        pass

    # ILS Loop
    current_codewords = set(best_codewords)
    
    while time.time() < deadline:
        # 1. Reduce
        current_codewords = reduce_codewords(current_codewords)
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            
        if not current_codewords:
            break
            
        # 2. Perturbation (Kick)
        # Remove 1 or 2 random codewords
        k = random.randint(1, 2)
        if len(current_codewords) > k:
            to_remove = random.sample(list(current_codewords), k)
            for c in to_remove:
                current_codewords.remove(c)
        
        # 3. Repair (Greedy Add)
        # Find all uncovered words
        covered = set()
        for c in current_codewords:
            covered.update(neighbors[c])
        uncovered = set(range(total_words)) - covered
        
        while uncovered and time.time() < deadline:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                if candidate in current_codewords:
                    continue
                count = 0
                for nb in neighbors[candidate]:
                    if nb in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_center = candidate
            
            if best_center == -1:
                break
            
            current_codewords.add(best_center)
            for nb in neighbors[best_center]:
                if nb in uncovered:
                    uncovered.remove(nb)

    # Final reduction on best result
    best_codewords = reduce_codewords(best_codewords)
    
    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result
