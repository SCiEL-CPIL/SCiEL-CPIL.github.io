"""Improved ternary covering code search for K_3(8,1).

Strategy:
  1. Start from the direct-sum construction (9 * 3^4 = 81 words? No, 9 * 3^(8-4)=729).
     Wait, the seed program produces 9 * 3^(n-4) = 9 * 3^4 = 729 words.
     The best known is 486. We need to reduce this significantly.
  2. Use a greedy set-cover style approach with local search / simulated annealing
     to remove redundant codewords while maintaining coverage.
  3. Key insight: A word is "redundant" if its sphere (17 words) is entirely covered by
     the union of spheres of other codewords. We can try to remove such words.
  4. To speed up, we track coverage counts: for each of the 6561 words, how many codewords
     cover it. A codeword c is redundant if every word in sphere(c) has coverage >= 2
     (i.e., covered by at least one other codeword).
  5. We iteratively remove redundant codewords. When no more can be removed, we try
     "swaps": remove a codeword and add a different one if it helps (but since we want
     to minimize, swaps that keep size same don't help directly; we need to find
     configurations where more codewords become redundant).
  6. Better: use a two-phase approach:
     Phase A: Greedy construction from scratch using weighted set cover (prioritize
              codewords that cover the most uncovered words).
     Phase B: Local search to remove redundancies.

Given the time constraint (150s), let's use:
  - Precompute all spheres (each codeword covers 17 words: itself + 8*2 = 17 neighbors).
  - Start with the 729-word construction.
  - Iteratively remove redundant codewords.
  - Use randomization/restarts to escape local optima.

Let's implement this with NumPy for speed.
"""
import time
import numpy as np
import itertools

# Precompute the mapping from word to index
# Word is a tuple of n integers in 0..q-1
# Index = sum(word[i] * q^(n-1-i))

def _word_to_index(word, q, n):
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx

def _index_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    total_words = q ** n
    # Precompute all spheres
    # sphere[c] = list of indices covered by codeword c
    # For radius 1: c itself + all words at Hamming distance 1
    # Number of neighbors: n * (q - 1)
    # For q=3, n=8: 8 * 2 = 16 neighbors + 1 = 17
    
    # Precompute all word indices and their spheres
    all_words = []
    for idx in range(total_words):
        all_words.append(_index_to_word(idx, q, n))
    
    # Precompute spheres: sphere[idx] = set of indices covered
    spheres = [None] * total_words
    for idx in range(total_words):
        word = all_words[idx]
        covered = set()
        covered.add(idx)
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = word[:]
                    new_word[pos] = val
                    new_idx = _word_to_index(new_word, q, n)
                    covered.add(new_idx)
        spheres[idx] = list(covered)
    
    # Start with the direct-sum construction
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            cw = h + list(tail)
            idx = _word_to_index(cw, q, n)
            initial_codewords.append(idx)
    
    initial_codewords = list(set(initial_codewords))
    
    def _is_covering(codeword_set):
        """Check if the set of codewords covers all words."""
        if len(codeword_set) == 0:
            return total_words == 0
        covered = np.zeros(total_words, dtype=np.uint8)
        for cw in codeword_set:
            for w in spheres[cw]:
                covered[w] = 1
        return np.all(covered)
    
    def _coverage_counts(codeword_set):
        """Compute coverage count for each word."""
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in codeword_set:
            for w in spheres[cw]:
                counts[w] += 1
        return counts
    
    def _find_redundant(codeword_list, counts):
        """Find codewords whose entire sphere is covered by others (counts >= 2 in sphere)."""
        redundant = []
        for cw in codeword_list:
            # Check if all words in sphere(cw) have count >= 2
            # If so, removing cw still leaves all covered
            is_redundant = True
            for w in spheres[cw]:
                if counts[w] < 2:
                    is_redundant = False
                    break
            if is_redundant:
                redundant.append(cw)
        return redundant
    
    # Phase 1: Start from initial construction, remove redundancies
    codeword_set = set(initial_codewords)
    best_size = len(codeword_set)
    best_codewords = list(codeword_set)
    
    # Iterative redundancy removal
    changed = True
    while changed:
        changed = False
        counts = _coverage_counts(codeword_set)
        redundant = _find_redundant(list(codeword_set), counts)
        for cw in redundant:
            codeword_set.remove(cw)
            changed = True
            if len(codeword_set) < best_size:
                best_size = len(codeword_set)
                best_codewords = list(codeword_set)
    
    # Phase 2: Local search with randomization
    # Try multiple restarts with perturbation
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Perturb: start from best, add some random codewords, then remove redundancies
        current_set = set(best_codewords)
        
        # Add some random codewords to break symmetry
        n_add = max(1, len(current_set) // 50)
        for _ in range(n_add):
            if time.time() > deadline:
                break
            random_idx = np.random.randint(0, total_words)
            current_set.add(int(random_idx))
        
        # Remove redundancies
        changed = True
        while changed:
            changed = False
            counts = _coverage_counts(current_set)
            redundant = _find_redundant(list(current_set), counts)
            for cw in redundant:
                current_set.remove(cw)
                changed = True
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_codewords = list(current_set)
    
    # Phase 3: More aggressive local search
    # Try to find codeword swaps that enable more removals
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        current_set = set(best_codewords)
        counts = _coverage_counts(current_set)
        
        # Find codewords that are "almost redundant" - their sphere has exactly one word
        # with count == 1. If we can find another codeword that covers that word, we can remove.
        # This is more complex; let's try a different approach.
        
        # Try removing each codeword one at a time and see if we can re-add fewer
        # Actually, let's try: for each codeword, try removing it. If coverage breaks,
        # find the minimum set of new codewords to restore coverage.
        
        # Simpler: try random removal + greedy repair
        if len(current_set) == 0:
            break
        
        # Pick a random codeword to remove
        cw_list = list(current_set)
        random_idx = np.random.randint(0, len(cw_list))
        remove_cw = cw_list[random_idx]
        
        trial_set = set(current_set)
        trial_set.remove(remove_cw)
        
        # Check if still covering
        if _is_covering(trial_set):
            best_size = len(trial_set)
            best_codewords = list(trial_set)
            current_set = trial_set
        else:
            # Find uncovered words
            counts = _coverage_counts(trial_set)
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                # Shouldn't happen if _is_covering returned False
                best_size = len(trial_set)
                best_codewords = list(trial_set)
                current_set = trial_set
            else:
                # Greedily add codewords to cover uncovered words
                for unc in uncovered:
                    # Find codeword that covers unc and maximizes new coverage
                    best_cw = None
                    best_gain = -1
                    for candidate in range(total_words):
                        if candidate in trial_set:
                            continue
                        gain = 0
                        for w in spheres[candidate]:
                            if counts[w] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_cw = candidate
                    if best_cw is not None and best_gain > 0:
                        trial_set.add(best_cw)
                        # Update counts
                        for w in spheres[best_cw]:
                            counts[w] += 1
                # After repair, remove redundancies
                changed = True
                while changed:
                    changed = False
                    redundant = _find_redundant(list(trial_set), counts)
                    for cw in redundant:
                        trial_set.remove(cw)
                        for w in spheres[cw]:
                            counts[w] -= 1
                        changed = True
                
                if len(trial_set) < best_size:
                    best_size = len(trial_set)
                    best_codewords = list(trial_set)
                    current_set = trial_set
    
    # Final verification
    if not _is_covering(set(best_codewords)):
        # Fallback to initial construction
        return [all_words[idx] for idx in initial_codewords]
    
    return [all_words[idx] for idx in best_codewords]
