import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses a hybrid approach:
    1. Start from the direct sum of the ternary tetracode (81 words for 3^6, radius 1).
    2. Perform intensive local search to remove redundant codewords and swap for better packing.
    3. Uses bitmasks for fast coverage checks and repair.
    """
    start_time = time.time()
    # 120s limit. Keep a safety margin. The evaluator kills at 150s, but we should aim for <120s.
    # Let's use 115s as deadline to be safe.
    deadline = start_time + 115.0

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
    # For radius=1, neighbors are the word itself and all words at distance 1.
    # Size of ball: 1 + n*(q-1). For q=3, n=6, radius=1: 1 + 6*2 = 13.
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
        # For this task (radius=1), we don't need radius 2 neighbors.
        # If radius > 1, this would need to be expanded, but the task specifies radius=1 for the main case.
        # The contract says "radius" is passed, but the specific task is K_3(6,1).
        # We assume radius=1 for the optimization logic. If radius != 1, we might fall back to a simpler greedy.
        
    # Build neighbor bitmasks for fast coverage checks
    # Python integers are arbitrary precision, so this is efficient for 729 bits.
    def build_mask(indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        return mask

    neighbor_masks = [build_mask(neighbors[i]) for i in range(total_words)]
    full_mask = (1 << total_words) - 1

    # Initialize with the direct sum of the tetracode
    # Tetracode: 9 words of length 4.
    # Direct sum with 2 more coordinates: 9 * 3 * 3 = 81 words.
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
        # Fallback: Greedy construction from scratch
        uncovered_mask = full_mask
        while uncovered_mask:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                count = bin(uncovered_mask & neighbor_masks[candidate]).count('1')
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            initial_codewords.add(best_center)
            uncovered_mask &= ~neighbor_masks[best_center]

    best_codewords = set(initial_codewords)
    best_size = len(best_codewords)

    def get_coverage_mask(codewords):
        mask = 0
        for c in codewords:
            mask |= neighbor_masks[c]
        return mask

    def compute_coverage_counts(codewords):
        """
        Returns an array where counts[w] is the number of codewords covering w.
        """
        counts = np.zeros(total_words, dtype=np.int32)
        for c in codewords:
            for nb in neighbors[c]:
                counts[nb] += 1
        return counts

    def is_redundant(c, codewords, counts):
        """
        Check if codeword c is redundant given current coverage counts.
        c is redundant if every word in its ball is covered by at least one other codeword.
        i.e., for all nb in neighbors[c], counts[nb] >= 2.
        """
        for nb in neighbors[c]:
            if counts[nb] <= 1:
                return False
        return True

    def reduce_codewords(codewords):
        """
        Iteratively remove redundant codewords until no more can be removed.
        """
        changed = True
        while changed and time.time() < deadline:
            changed = False
            counts = compute_coverage_counts(codewords)
            for c in list(codewords):
                if is_redundant(c, codewords, counts):
                    codewords.remove(c)
                    changed = True
                    break  # Restart loop to recompute counts
        return codewords

    def greedy_repair(codewords):
        """
        Add codewords to cover any uncovered words.
        """
        covered = get_coverage_mask(codewords)
        uncovered = full_mask & ~covered
        while uncovered and time.time() < deadline:
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                if candidate in codewords:
                    continue
                count = bin(uncovered & neighbor_masks[candidate]).count('1')
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1:
                break
            codewords.add(best_center)
            uncovered &= ~neighbor_masks[best_center]
        return codewords

    def find_least_critical(codewords, k=1):
        """
        Find the k least critical codewords.
        Criticality is measured by the minimum coverage count in their ball.
        Lower minimum count -> more critical.
        We want the ones with the HIGHEST minimum count (least critical).
        """
        if len(codewords) <= k:
            return list(codewords)
        
        counts = compute_coverage_counts(codewords)
        criticality = []
        for c in codewords:
            min_count = min(counts[nb] for nb in neighbors[c])
            criticality.append((min_count, c))
        
        # Sort by descending min_count (highest min_count is least critical)
        criticality.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in criticality[:k]]

    # Main Search Loop
    current_codewords = set(best_codewords)
    
    # First, reduce the initial solution
    current_codewords = reduce_codewords(current_codewords)
    if len(current_codewords) < best_size:
        best_size = len(current_codewords)
        best_codewords = set(current_codewords)

    # Iterated Local Search
    while time.time() < deadline - 5:
        # 1. Perturbation
        # Strategy: Remove 1-2 least critical codewords, then repair.
        # Sometimes swap a least critical one with a random neighbor to escape local minima.
        
        perturbation_type = random.random()
        
        if perturbation_type < 0.6:
            # Simple removal
            k = random.randint(1, 2)
            if len(current_codewords) > k:
                least_critical = find_least_critical(current_codewords, k)
                for c in least_critical:
                    current_codewords.remove(c)
        elif perturbation_type < 0.8:
            # Swap
            if len(current_codewords) > 1:
                least_critical = find_least_critical(current_codewords, 1)
                to_remove = least_critical[0]
                # Pick a random word from the ball of to_remove that is not in the code
                candidates = [nb for nb in neighbors[to_remove] if nb not in current_codewords]
                if candidates:
                    to_add = random.choice(candidates)
                    current_codewords.remove(to_remove)
                    current_codewords.add(to_add)
                else:
                    current_codewords.remove(to_remove)
        else:
            # Random removal of 1-3
            k = random.randint(1, 3)
            if len(current_codewords) > k:
                to_remove = random.sample(list(current_codewords), k)
                for c in to_remove:
                    current_codewords.remove(c)
        
        # 2. Repair
        current_codewords = greedy_repair(current_codewords)
        
        # 3. Reduce
        current_codewords = reduce_codewords(current_codewords)
        
        # 4. Update best
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = set(current_codewords)
            
            # If we hit the theoretical lower bound or a very good size, we can stop earlier?
            # Lower bound is 71. If we reach 71, we are done.
            if best_size <= 71:
                break

    # Final Polish: Try 2-swap moves (remove 2, add 1) on the best solution
    # This can sometimes reduce the size by 1.
    final_codewords = set(best_codewords)
    
    if time.time() < deadline - 2:
        # Try to find a 2-for-1 improvement
        # Iterate through pairs of codewords
        codeword_list = list(final_codewords)
        random.shuffle(codeword_list)
        
        improved = False
        for i in range(len(codeword_list)):
            if time.time() > deadline - 2:
                break
            if improved:
                break
            c1 = codeword_list[i]
            if c1 not in final_codewords:
                continue
            for j in range(i + 1, len(codeword_list)):
                if time.time() > deadline - 2:
                    break
                c2 = codeword_list[j]
                if c2 not in final_codewords:
                    continue
                
                # Tentatively remove c1 and c2
                test_set = set(final_codewords)
                test_set.remove(c1)
                test_set.remove(c2)
                
                covered = get_coverage_mask(test_set)
                uncovered = full_mask & ~covered
                
                if not uncovered:
                    # We removed 2 and didn't lose coverage. This is a 2-for-0 move (reduction by 2).
                    # This is even better!
                    final_codewords = test_set
                    improved = True
                    break
                
                # Try to find a single codeword that covers all uncovered
                best_add = -1
                for candidate in range(total_words):
                    if candidate in test_set:
                        continue
                    # Check if candidate covers all uncovered
                    if (uncovered & ~neighbor_masks[candidate]) == 0:
                        best_add = candidate
                        break
                
                if best_add != -1:
                    # We removed 2 and added 1. Net reduction of 1.
                    final_codewords = test_set
                    final_codewords.add(best_add)
                    improved = True
                    break
        
        if len(final_codewords) < best_size:
            best_size = len(final_codewords)
            best_codewords = set(final_codewords)

    # Convert back to list of lists
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result
