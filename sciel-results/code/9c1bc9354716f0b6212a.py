import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses Iterated Local Search (Greedy + Perturbation + Repair) to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
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

    # Precompute neighbors for all words as lists
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
    # For radius >= 2, add second-order neighbors
    if radius >= 2:
        for i in range(total_words):
            first_order = set(neighbors[i])
            for j in first_order:
                for k in neighbors[j]:
                    if k not in neighbors[i]:
                        neighbors[i].append(k)
        # Remove duplicates and sort
        for i in range(total_words):
            neighbors[i] = list(set(neighbors[i]))

    # Build neighbor bitmasks for fast coverage checks
    # Use Python integers as bitmasks (arbitrary precision)
    def build_mask(indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        return mask

    neighbor_masks = [build_mask(neighbors[i]) for i in range(total_words)]
    
    # Precompute: for each word w, which codeword indices cover w
    # Actually, we'll compute coverage on the fly using masks

    # Initialize with a known good construction: Direct sum of tetracode
    initial_codewords = set()
    if q == 3 and n == 6 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    initial_codewords.add(idx)
    else:
        # Fallback: Greedy construction from scratch with symmetry-breaking tie-breaking
        uncovered_mask = (1 << total_words) - 1
        while uncovered_mask:
            best_center = -1
            best_count = -1
            best_weight = float('inf')  # For tie-breaking: prefer lower integer value
            for candidate in range(total_words):
                count = bin(uncovered_mask & neighbor_masks[candidate]).count('1')
                if count > best_count or (count == best_count and candidate < best_weight):
                    best_count = count
                    best_center = candidate
                    best_weight = candidate
            if best_center == -1:
                break
            initial_codewords.add(best_center)
            uncovered_mask &= ~neighbor_masks[best_center]

    best_codewords = set(initial_codewords)
    best_size = len(best_codewords)
    
    def get_coverage_mask(codewords):
        """Calculate the coverage bitmask given a set of codewords."""
        mask = 0
        for c in codewords:
            mask |= neighbor_masks[c]
        return mask

    def reduce_codewords_bitmask(codewords, max_iterations=50):
        """Remove redundant codewords using bitmask coverage. Returns the reduced set."""
        changed = True
        iterations = 0
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            if time.time() > deadline:
                break
            # Compute total coverage once per pass
            total_cover = get_coverage_mask(codewords)
            # Try removing each codeword in deterministic order for stability
            for c in sorted(codewords):
                if time.time() > deadline:
                    break
                # Check if all neighbors of c are still covered by others
                cover_without_c = total_cover & ~neighbor_masks[c]
                if (cover_without_c & neighbor_masks[c]) == neighbor_masks[c]:
                    codewords.remove(c)
                    total_cover = get_coverage_mask(codewords)  # Update after removal
                    changed = True
        return codewords

    def greedy_repair(codewords):
        """Add codewords greedily to cover all uncovered words. Returns modified set."""
        covered = get_coverage_mask(codewords)
        uncovered = ((1 << total_words) - 1) & ~covered
        while uncovered:
            if time.time() > deadline:
                break
            best_center = -1
            best_count = -1
            best_weight = float('inf')
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                count = bin(uncovered & neighbor_masks[candidate]).count('1')
                if count > best_count or (count == best_count and candidate < best_weight):
                    best_count = count
                    best_center = candidate
                    best_weight = candidate
            if best_center == -1:
                break
            codewords.add(best_center)
            uncovered &= ~neighbor_masks[best_center]
        return codewords

    # Main ILS Loop
    current_codewords = set(best_codewords)
    il_iteration = 0
    
    while time.time() < deadline:
        il_iteration += 1
        # 1. Reduce
        current_codewords = reduce_codewords_bitmask(current_codewords)
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            
        if not current_codewords:
            break
            
        # 2. Perturbation (Kick)
        perturbation_type = random.random()
        if perturbation_type < 0.35:
            # Remove 1-2 random codewords (milder kick)
            k = random.randint(1, 2)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
        elif perturbation_type < 0.65:
            # Swap: remove one, add one from neighborhood of a random codeword
            if len(current_codewords) > 1:
                to_remove = random.choice(list(current_codewords))
                current_codewords.remove(to_remove)
                # Add a nearby word (within distance 1) not in codewords
                nearby = [nb for nb in neighbors[to_remove] if nb not in current_codewords]
                if nearby:
                    to_add = random.choice(nearby)
                    current_codewords.add(to_add)
                else:
                    # Fallback: random word
                    candidates = [i for i in range(total_words) if i not in current_codewords]
                    if candidates:
                        to_add = random.choice(candidates)
                        current_codewords.add(to_add)
        else:
            # Remove 2-3 random codewords
            k = random.randint(2, 3)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
        
        # 3. Repair (Greedy Add)
        current_codewords = greedy_repair(current_codewords)

    # Final polish: try harder to reduce the best solution
    final_codewords = set(best_codewords)
    polish_deadline = deadline - 3.0
    
    # Pass 1: Multiple reduction passes with different orderings
    for _ in range(100):
        if time.time() > polish_deadline:
            break
        shuffled = list(final_codewords)
        random.shuffle(shuffled)
        changed = False
        total_cover = get_coverage_mask(final_codewords)
        for c in shuffled:
            if c not in final_codewords:
                continue
            cover_without_c = total_cover & ~neighbor_masks[c]
            if (cover_without_c & neighbor_masks[c]) == neighbor_masks[c]:
                final_codewords.remove(c)
                total_cover = get_coverage_mask(final_codewords)
                changed = True
        if not changed:
            break
        if len(final_codewords) < best_size:
            best_size = len(final_codewords)
            best_codewords = set(final_codewords)
    
    # Pass 2: Try 2-swap moves (remove 2, add 1) - limited to avoid timeout
    if time.time() < polish_deadline:
        for _ in range(10):
            if time.time() > polish_deadline:
                break
            found_improvement = False
            codeword_list = list(final_codewords)
            random.shuffle(codeword_list)
            # Limit pairs to check
            max_pairs = 200
            pairs_checked = 0
            for i in range(len(codeword_list)):
                if pairs_checked >= max_pairs:
                    break
                for j in range(i + 1, len(codeword_list)):
                    if pairs_checked >= max_pairs:
                        break
                    pairs_checked += 1
                    c1, c2 = codeword_list[i], codeword_list[j]
                    if c1 not in final_codewords or c2 not in final_codewords:
                        continue
                    test_set = set(final_codewords)
                    test_set.remove(c1)
                    test_set.remove(c2)
                    covered = get_coverage_mask(test_set)
                    uncovered = ((1 << total_words) - 1) & ~covered
                    if uncovered:
                        uncovered_count = bin(uncovered).count('1')
                        best_add = -1
                        for candidate in range(total_words):
                            if candidate in test_set:
                                continue
                            count = bin(uncovered & neighbor_masks[candidate]).count('1')
                            if count == uncovered_count:
                                best_add = candidate
                                break
                        if best_add != -1:
                            test_set.add(best_add)
                            final_cover = get_coverage_mask(test_set)
                            if final_cover == (1 << total_words) - 1:
                                final_codewords = test_set
                                found_improvement = True
                                break
                if found_improvement:
                    if len(final_codewords) < best_size:
                        best_size = len(final_codewords)
                        best_codewords = set(final_codewords)
                    break

    # Convert best_codewords (set of ints) to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result
