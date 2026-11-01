# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    if n == 0:
        return [[]]

    # Specific case: q=3, n=6, r=1
    # Known optimal: 73. Direct sum: 81.
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Fallback for other cases
    return _generic_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = q ** n # 729
    
    # Precompute neighbors for all words as bitsets (Python ints)
    # Index of word w: w[0]*3^5 + w[1]*3^4 + ... + w[5]
    # We need a mapping from word tuple to index, and index to word tuple.
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor bitsets
    # neighbor_mask[i] is an integer where bit j is 1 if all_words[j] is in N(all_words[i])
    neighbor_masks = [0] * total_words
    
    for i, w in enumerate(all_words):
        mask = 1 << i # The word itself
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                idx = word_to_idx[tuple(new_w)]
                mask |= (1 << idx)
        neighbor_masks[i] = mask

    # Helper to get coverage mask of a set of indices
    def get_coverage_mask(indices):
        mask = 0
        for idx in indices:
            mask |= neighbor_masks[idx]
        return mask

    def is_valid(indices):
        mask = get_coverage_mask(indices)
        # Check if mask covers all 729 words
        # Full mask is (1 << 729) - 1
        # This is a big integer, but Python handles it.
        # Alternatively, check popcount.
        # popcount of mask should be 729.
        # int.bit_count() is available in Python 3.8+? 
        # Let's use bin(mask).count('1') or a lookup table if needed.
        # 729 bits is small enough for bin().
        return mask.bit_count() == total_words

    def get_size(indices):
        return len(indices)

    # Initial Solution: Direct Sum Code (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = tuple(h + list(tail))
            init_indices.append(word_to_idx[word])
    
    # Ensure init is valid (it should be)
    if not is_valid(init_indices):
        # Fallback to greedy if direct sum fails (shouldn't happen)
        return _greedy_fallback(q, n, radius, deadline)

    best_indices = set(init_indices)
    best_size = len(best_indices)
    
    # PSO / Population Based Local Search
    population_size = 30
    max_iterations = 5000 # Adjust based on time
    
    # Particle: set of indices
    # Personal Best: set of indices
    # Global Best: best_indices
    
    # Initialize population
    population = [set(init_indices) for _ in range(population_size)]
    personal_bests = [set(init_indices) for _ in range(population_size)]
    
    # Add some random diverse particles
    for i in range(population_size):
        if i > 0:
            # Start with a random subset of size ~80
            random_subset = set(random.sample(range(total_words), 80))
            # Repair: add words until valid
            while not is_valid(random_subset):
                # Find an uncovered word
                cov_mask = get_coverage_mask(random_subset)
                # Find an index not in cov_mask
                # This is slow if done naively.
                # Better: Pick a random word not in the set and add it?
                # Or pick a word that covers the most uncovered?
                # For simplicity, just add random words until valid.
                # This might take a while if 80 is too small.
                # Let's start with 90 random words.
                pass # Skip complex repair for init, just use perturbed direct sum
            
            # Perturb direct sum
            perturbed = set(init_indices)
            # Remove some random words
            for _ in range(5):
                if perturbed:
                    rem = random.choice(list(perturbed))
                    perturbed.remove(rem)
            # Add some random words
            for _ in range(5):
                add = random.randint(0, total_words - 1)
                perturbed.add(add)
            
            # Repair perturbed
            if not is_valid(perturbed):
                 # Simple repair: add random words until valid
                 while not is_valid(perturbed) and len(perturbed) < total_words:
                     add = random.randint(0, total_words - 1)
                     perturbed.add(add)
            if is_valid(perturbed):
                population[i] = perturbed
                personal_bests[i] = perturbed

    # Main Loop
    for iter in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        for p in range(population_size):
            # Update Particle p
            current = population[p]
            p_best = personal_bests[p]
            
            # PSO Velocity-like update:
            # For each word in the union of current, p_best, g_best, decide to include or exclude.
            # This is expensive. Instead, do local moves guided by bests.
            
            # 1. Try to remove a word from current that is NOT in p_best or g_best
            removable = [w for w in current if w not in p_best and w not in best_indices]
            if removable:
                w = random.choice(removable)
                new_current = set(current)
                new_current.remove(w)
                if is_valid(new_current):
                    population[p] = new_current
                    if len(new_current) < len(p_best):
                        personal_bests[p] = new_current
                    if len(new_current) < len(best_indices):
                        best_indices = new_current
                        best_size = len(best_indices)
                else:
                    # Try to swap with a word from p_best or g_best?
                    pass
            
            # 2. Try to remove a word that is in p_best but not g_best? No, p_best is better than current.
            
            # 3. Try to add a word from p_best or g_best that is not in current?
            # This increases size, usually bad unless it allows removals later.
            # Let's stick to removal and swaps.
            
            # 4. Swap: Remove w1 from current, add w2.
            # w2 should be from p_best or g_best to move towards them.
            candidates_to_add = [w for w in (p_best | best_indices) if w not in current]
            if candidates_to_add and current:
                w2 = random.choice(candidates_to_add)
                w1 = random.choice(list(current))
                new_current = set(current)
                new_current.remove(w1)
                new_current.add(w2)
                if is_valid(new_current):
                    population[p] = new_current
                    if len(new_current) < len(p_best):
                        personal_bests[p] = new_current
                    if len(new_current) < len(best_indices):
                        best_indices = new_current
                        best_size = len(best_indices)

            # 5. Pure Local Search on current: Try to remove any word
            # This is O(N^2) per particle if done naively.
            # Optimization: Only try removals if we haven't improved in a while?
            # Let's do a few random removal attempts.
            for _ in range(3):
                if not current:
                    break
                w = random.choice(list(current))
                new_current = set(current)
                new_current.remove(w)
                if is_valid(new_current):
                    population[p] = new_current
                    if len(new_current) < len(p_best):
                        personal_bests[p] = new_current
                    if len(new_current) < len(best_indices):
                        best_indices = new_current
                        best_size = len(best_indices)
                    current = new_current

    # Final Polish on best_indices
    # Try to remove any redundant words
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        for w in list(best_indices):
            new_set = set(best_indices)
            new_set.remove(w)
            if is_valid(new_set):
                best_indices = new_set
                changed = True
                break # Restart loop as indices changed

    return [list(all_words[i]) for i in best_indices]

def _greedy_fallback(q, n, radius, deadline):
    # Simple greedy
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    def get_neighbors(c):
        neighbors = {tuple(c)}
        for i in range(n):
            orig = c[i]
            for val in range(q):
                if val == orig: continue
                new_c = c.copy()
                new_c[i] = val
                neighbors.add(tuple(new_c))
        return neighbors

    while uncovered and time.time() < deadline:
        best_word = None
        best_count = -1
        for w in all_words:
            if w in uncovered:
                nb = get_neighbors(w)
                cnt = len(nb & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_word = w
        if best_word is None:
            break
        code.append(list(best_word))
        uncovered.difference_update(get_neighbors(best_word))
        
    return code

def _generic_construct(q, n, radius, deadline):
    # Try direct sum for q=3
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Fallback greedy
    return _greedy_fallback(q, n, radius, deadline)
# EVOLVE-BLOCK-END
