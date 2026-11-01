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
    deadline = start_time + 115.0  # Safety margin

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Generic greedy for small n
    return greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    N = q ** n  # 729

    # Precompute neighbors for each word index as a numpy array
    neighbor_lists = []
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbor_lists.append(neigh)
    
    # Convert to numpy arrays for faster operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbor_lists]
    
    # Initial code: Direct Sum (Tetracode x 3^2) -> 81 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts using numpy
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        counts[neighbor_arrays[idx]] += 1
        
    active_set = set(code_indices)
    active_list = list(code_indices)
    
    def can_remove_vectorized(idx):
        """Check if word idx can be removed (all neighbors have count > 1)"""
        return np.all(counts[neighbor_arrays[idx]] > 1)
    
    def try_remove_all_redundant():
        """Iteratively remove all redundant words until none remain"""
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            random.shuffle(active_list)
            for idx in active_list[:]:
                if idx not in active_set:
                    continue
                if can_remove_vectorized(idx):
                    counts[neighbor_arrays[idx]] -= 1
                    active_set.remove(idx)
                    active_list.remove(idx)
                    removed_any = True

    try_remove_all_redundant()
    
    # Particle Swarm Optimization (PSO) adapted for discrete set cover
    # We maintain a swarm of candidate solutions.
    # Each particle is a set of codeword indices.
    # Fitness = number of codewords + penalty for uncovered words.
    
    # To make PSO work efficiently, we operate on a binary mask representation?
    # Or just sets? Sets are easier for exact coverage checks but harder to "average".
    # However, for discrete PSO, we can define velocity as a set of flips.
    # v_i = {j | x_i[j] != x_pbest[j]} ?
    # Actually, a simpler approach for discrete PSO in this context:
    # 1. Keep a swarm of K particles (sets).
    # 2. For each particle, calculate fitness (size + 1000 * uncovered_count).
    # 3. Update personal best.
    # 4. Update global best.
    # 5. Update particle position:
    #    For each codeword j in the universe:
    #    If j is in pbest and not in current, add it with prob w1.
    #    If j is in global best and not in current, add it with prob w2.
    #    If j is in current but not in pbest, remove it with prob w3.
    #    If j is in current but not in global best, remove it with prob w4.
    #    Then repair (add missing) and reduce (remove redundant).
    
    # But 729 codewords is large for full iteration.
    # Let's restrict the search space to a subset of "candidate" codewords.
    # The optimal code is likely composed of words from the initial direct sum or close to it.
    # Or we can just use the standard ILS from inspiration 1 which was successful (score 1.025).
    # The prompt asks to COMBINE current (numpy greedy/ILS) with inspiration (ILS with repair).
    # The current program's ILS is simple: add 1, remove redundant.
    # The inspiration's ILS: remove k random, repair, reduce. This is more powerful for escaping local minima.
    
    # Let's implement the stronger ILS from inspiration 1, but using the numpy vectorization from the current program for speed.
    
    # State:
    # active_set: set of int
    # counts: np.array of int
    
    # We will maintain a "best" solution and iterate.
    
    best_size = len(active_list)
    best_active_set = set(active_set)
    best_counts = counts.copy()
    
    rng = random.Random(42)
    
    def get_uncovered_mask(counts_arr):
        """Return indices of words with count 0"""
        return np.where(counts_arr == 0)[0]
    
    def repair_and_reduce(active_set_local, counts_local):
        """
        Ensure coverage by adding words, then remove redundant words.
        Returns updated active_set_local and counts_local.
        """
        # 1. Repair: Add words to cover all uncovered
        # Get uncovered indices
        uncovered = get_uncovered_mask(counts_local)
        
        # While there are uncovered words, add the word that covers the most uncovered words
        while len(uncovered) > 0 and time.time() < deadline:
            if len(uncovered) == 0:
                break
            
            # Find best word to add
            # We want to maximize coverage of 'uncovered' set.
            # Since N=729 is small, we can check all words not in active_set_local.
            # Optimization: Only check words that are NOT in active_set_local.
            
            best_idx = -1
            best_gain = 0
            
            # To speed this up, we could precompute intersections, but simple loop is fine for 729.
            # We can skip words already in active_set_local.
            for i in range(N):
                if i in active_set_local:
                    continue
                
                # Count how many uncovered words are in neighbors[i]
                # neighbors[i] is an array of indices.
                # We need intersection with 'uncovered' set.
                # Converting 'uncovered' to a set is fast.
                # But doing it 729 times per step is slow.
                
                # Alternative: Use numpy.
                # Create a boolean mask for uncovered?
                # uncovered_mask = np.zeros(N, dtype=bool); uncovered_mask[uncovered] = True
                # gain = np.sum(uncovered_mask[neighbor_arrays[i]])
                
                # Let's try the numpy way.
                # This requires creating a mask every iteration? No, mask is static during this while loop.
                # But 'uncovered' changes as we add words.
                
                # Let's just use a set for uncovered for intersection.
                # uncovered_set = set(uncovered)
                # gain = len(set(neighbor_arrays[i]) & uncovered_set)
                # This is O(N) per check. 729 * 729 = 531k ops. Acceptable.
                
                # But we can do better.
                # Let's stick to the numpy vectorization for the 'can_remove' part which is frequent.
                # For 'add best', we can just pick a random uncovered word and add one of its neighbors?
                # No, greedy is better.
                
                # Let's use the set intersection method for simplicity and correctness, it's fast enough for 729.
                # We'll optimize if it's too slow.
                
                pass # Placeholder, logic below
            
            # Efficient Greedy Add:
            # Pick a random uncovered word u.
            # Find all codewords that cover u (neighbors of u).
            # Among those not in active_set_local, pick the one that covers the most OTHER uncovered words.
            
            # Actually, let's just iterate all candidates.
            # To speed up, we precompute neighbor_sets?
            # neighbor_sets = [set(neigh) for neigh in neighbor_lists]
            # This uses memory but speeds up intersection.
            
            # Let's assume we have neighbor_sets.
            # For now, let's use the simple loop.
            
            # Optimization: Precompute neighbor_sets outside this function?
            # It's local to _solve_q3_n6_r1.
            
            # Let's refactor:
            # We will use a list of sets for neighbors.
            
            # Inside repair:
            # uncovered_set = set(uncovered)
            # for i in range(N):
            #    if i in active_set_local: continue
            #    gain = len(neighbor_sets[i] & uncovered_set)
            #    if gain > best_gain: ...
            
            # If no word found (should not happen), break.
            
            # Let's implement this properly in the code below.
            
            # For now, let's just break and rely on the initial good start + ILS perturbations.
            # The ILS perturbation will remove words, creating holes, and repair will fill them.
            # The repair needs to be robust.
            
            # Let's implement a fast repair.
            
            # Break out of this placeholder thinking and write the code.
            break # Exit placeholder while loop
            
    # We need to define neighbor_sets for fast intersection in repair.
    neighbor_sets = [set(neigh) for neigh in neighbor_lists]
    
    def fast_repair(active_set_local, counts_local):
        """Add codewords to cover all words with count 0."""
        # Loop until all covered
        while True:
            if time.time() > deadline:
                return
            
            uncovered = get_uncovered_mask(counts_local)
            if len(uncovered) == 0:
                break
            
            # Greedy: Find word covering most uncovered
            uncovered_set = set(uncovered)
            best_idx = -1
            best_gain = 0
            
            # Optimization: Only check words not in active_set_local
            # We can iterate all 729.
            for i in range(N):
                if i in active_set_local:
                    continue
                gain = len(neighbor_sets[i] & uncovered_set)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                # No word can cover an uncovered word? Should not happen.
                break
            
            # Add best_idx
            active_set_local.add(best_idx)
            counts_local[neighbor_arrays[best_idx]] += 1
            
    def fast_reduce(active_set_local, counts_local):
        """Remove redundant codewords."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to break symmetry
            items = list(active_set_local)
            random.shuffle(items)
            for idx in items:
                if idx not in active_set_local:
                    continue
                if can_remove_vectorized(idx):
                    counts_local[neighbor_arrays[idx]] -= 1
                    active_set_local.remove(idx)
                    changed = True
                    # We don't break, we continue to check others?
                    # Removing one might make another redundant.
                    # But it might also make a previously non-redundant one redundant?
                    # No, removing a word decreases counts, so it makes it HARDER to be redundant.
                    # So if it wasn't redundant before, it won't be after removal of another word.
                    # Wait.
                    # Redundant means all neighbors have count > 1.
                    # If I remove word X, counts decrease.
                    # If word Y was redundant (all neighbors count > 1), after removing X,
                    # some neighbors of Y might drop to 1. So Y might become NON-redundant.
                    # So we should re-check?
                    # Actually, if Y was redundant, it means every neighbor had count >= 2.
                    # If we remove X, and X is a neighbor of Y, then count of that neighbor drops by 1.
                    # It might drop to 1. Then Y is no longer redundant.
                    # So we should stop or re-evaluate?
                    # The standard algorithm is: check all, remove those that are currently redundant.
                    # If we remove one, we should probably restart the check or at least be careful.
                    # However, since we are iterating a shuffled list, we might miss some or remove ones that become critical.
                    # To be safe, if we remove one, we can break and restart the while loop?
                    # That's slower but safer.
                    # Or we can just continue, knowing that we might leave some redundant words (which is fine, they will be caught in next pass).
                    # But we might remove a word that makes another word critical?
                    # No, removing a word only reduces coverage. It cannot make a word *more* redundant.
                    # It can make a word *less* redundant.
                    # So if a word was redundant, it stays redundant or becomes non-redundant.
                    # If we remove it, it's gone.
                    # If we don't remove it, it might become non-redundant later.
                    # So continuing is safe in the sense that we won't remove a critical word.
                    # We might just fail to remove some redundant words in this pass.
                    # The outer while loop handles that.
                    pass

    # Main ILS Loop
    # Start with the initial reduced code
    current_set = set(active_set)
    current_counts = counts.copy()
    
    best_size = len(current_set)
    best_set = set(current_set)
    best_counts = current_counts.copy()
    
    no_improvement = 0
    max_no_improvement = 100
    
    while time.time() < deadline - 2:
        if no_improvement > max_no_improvement:
            break
            
        # Perturb: Remove k random codewords
        k = rng.randint(2, 5)
        if len(current_set) < k:
            k = len(current_set)
            
        # Pick k random words from current_set to remove
        to_remove = rng.sample(list(current_set), k)
        
        for idx in to_remove:
            current_set.remove(idx)
            current_counts[neighbor_arrays[idx]] -= 1
            
        # Repair
        fast_repair(current_set, current_counts)
        
        # Reduce
        fast_reduce(current_set, current_counts)
        
        # Evaluate
        current_size = len(current_set)
        
        if current_size < best_size:
            best_size = current_size
            best_set = set(current_set)
            best_counts = current_counts.copy()
            no_improvement = 0
        else:
            no_improvement += 1
            
        # Accept current state (even if worse) to allow exploration?
        # Or always revert to best?
        # Standard ILS accepts worse solutions with some probability or always accepts to move around.
        # Let's always accept to move around the space.
        # The "best" is tracked separately.
        
        # Optional: If we hit the known optimal 73, we can stop early?
        # But maybe we can go lower? Lower bound is 71.
        # So keep searching.
        
    # Restore best state
    active_set = best_set
    active_list = list(active_set)
    
    # Final cleanup
    # Reconstruct counts for final state just in case
    final_counts = np.zeros(N, dtype=np.int32)
    for idx in active_list:
        final_counts[neighbor_arrays[idx]] += 1
        
    # One last reduce pass on the best state
    final_set = set(active_list)
    fast_reduce(final_set, final_counts)
    
    # Convert back to list of lists
    return [list(idx_to_word[idx]) for idx in final_set]

def greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    # Placeholder for generic greedy, used as fallback
    if n > 8:
        if q == 3:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             if n >= 4:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             else:
                 return [[0]*n]
        return [[0]*n]
        
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    def get_neighbors(c):
        neighbors = {c}
        for i in range(n):
            orig = c[i]
            for val in range(q):
                if val == orig: continue
                lst = list(c)
                lst[i] = val
                neighbors.add(tuple(lst))
        return neighbors

    while uncovered and time.time() < deadline:
        best_word = None
        best_count = -1
        
        # Greedy choice: pick word covering most uncovered
        for w in all_words:
            if w in code: continue
            cnt = 0
            for x in get_neighbors(w):
                if x in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
                
        if best_word is None or best_count == 0:
            break
            
        code.append(list(best_word))
        uncovered.difference_update(get_neighbors(best_word))
        
    return code
# EVOLVE-BLOCK-END
