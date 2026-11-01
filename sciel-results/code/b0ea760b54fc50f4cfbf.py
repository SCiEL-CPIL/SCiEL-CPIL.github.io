import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Fallback for general case or if not the specific hard instance
    if not (q == 3 and n == 6 and radius == 1):
        # Use the seed logic for general fallback if possible
        # The seed logic works for q=3, n>=4.
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial valid code: all words
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres as numpy boolean matrix
    # spheres_arr[i, j] is True if word j is within radius of word i
    spheres_arr = np.zeros((N, N), dtype=bool)
    for i, w in enumerate(all_words):
        spheres_arr[i, i] = True
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                spheres_arr[i, j] = True

    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial code: Tetracode construction (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = tuple(h) + tuple(tail)
            initial_code_indices.append(word_to_idx[word])
    
    best_code_indices = initial_code_indices[:]
    best_size = len(best_code_indices)

    # Helper function to check if a set of indices is a valid cover
    # Returns True if valid, False otherwise
    def is_valid_cover(indices):
        if not indices:
            return False
        covered = np.zeros(N, dtype=bool)
        for idx in indices:
            covered |= spheres_arr[idx]
        return np.all(covered)

    # Helper function to remove redundant codewords
    # Returns the reduced list of indices
    def reduce_code(indices):
        # Start with the given indices
        current_indices = list(indices)
        
        # We can speed this up by maintaining a coverage count array
        # But for N=729 and code size ~80, simple boolean OR is fast enough
        # Let's use the coverage count method for robustness and speed in inner loops
        
        while True:
            # Check if we can remove any codeword
            removed_any = False
            # Create a copy to iterate over, as we might modify the list
            # Actually, we should check redundancy against the *other* codewords
            
            # Efficient redundancy check:
            # A codeword i is redundant if sphere(i) is subset of union(sphere(j) for j != i)
            # Equivalently, if we remove i, is the space still covered?
            
            # Let's maintain a count of how many codewords cover each word
            # This allows O(1) redundancy check: if a word is covered only by i, i is not redundant.
            # If all words in sphere(i) are covered by >= 2 codewords, i is redundant.
            
            # However, maintaining counts dynamically is tricky with swaps.
            # Given the small size (80-90 words), let's just do the boolean check for simplicity first, 
            # but optimize if it's too slow.
            
            # Actually, for ~80 words, checking each word's redundancy by re-computing coverage 
            # of the sphere of that word is 80 * (729 + 12) operations.
            # 80 * 741 ~ 60k ops. This is very fast.
            
            for idx in current_indices:
                # Check if removing idx leaves the code valid
                # To do this, we need to know if every word in sphere(idx) is covered by some other codeword.
                
                # Optimization: Pre-compute "unique coverage" is hard.
                # Let's just try to remove it and check validity? 
                # No, that's O(N) per check.
                
                # Better: 
                # Let S be the set of current indices.
                # i is redundant if for all w in sphere(i), there exists j in S\{i} such that w in sphere(j).
                
                # Let's compute the union of spheres of all *other* codewords?
                # That's expensive to do from scratch each time.
                
                # Alternative: 
                # Maintain `coverage_counts` array.
                # When adding/removing, update counts.
                # A codeword i is redundant if min(coverage_counts[sphere(i)]) >= 2.
                
                # Let's implement the count-based reduction.
                pass

        return current_indices

    # Let's implement a simpler, robust reduction using counts
    def reduce_code_with_counts(indices):
        # Initialize coverage counts
        counts = np.zeros(N, dtype=np.int32)
        for idx in indices:
            counts[spheres_arr[idx]] += 1
        
        current_indices = list(indices)
        changed = True
        while changed:
            changed = False
            # Iterate through a copy of indices
            for idx in current_indices:
                # Check if idx is redundant
                # If min(counts[sphere(idx)]) >= 2, then idx is redundant
                sphere_mask = spheres_arr[idx]
                min_count = np.min(counts[sphere_mask])
                if min_count >= 2:
                    # Remove idx
                    current_indices.remove(idx)
                    counts[sphere_mask] -= 1
                    changed = True
                    break # Restart loop as indices changed
        return current_indices

    # Local search loop
    # We start with the tetracode code and try to improve it
    current_indices = reduce_code_with_counts(best_code_indices)
    current_size = len(current_indices)
    
    if current_size < best_size:
        best_size = current_size
        best_code_indices = current_indices[:]

    rng = np.random.default_rng(42)
    
    # Time budget for local search
    # We will perform random swaps and deletions
    while time.time() < deadline:
        # Try a random move
        # Move type 1: Swap a codeword with a random other word
        # Move type 2: Add a random word, then reduce
        # Move type 3: Remove a random word (if valid), then try to add back better ones? No, just reduce.
        
        move_type = rng.integers(0, 3)
        trial_indices = current_indices[:]
        
        if move_type == 0 and len(current_indices) > 0:
            # Swap
            idx_to_remove = current_indices[rng.integers(0, len(current_indices))]
            new_idx = rng.integers(0, N)
            if new_idx not in current_indices:
                trial_indices.remove(idx_to_remove)
                trial_indices.append(new_idx)
        elif move_type == 1:
            # Add random
            new_idx = rng.integers(0, N)
            if new_idx not in current_indices:
                trial_indices.append(new_idx)
        elif move_type == 2:
            # Remove random (if valid)
            if len(current_indices) > 1:
                idx_to_remove = current_indices[rng.integers(0, len(current_indices))]
                trial_indices.remove(idx_to_remove)
                # Check if still valid
                # We can check validity quickly
                covered = np.zeros(N, dtype=bool)
                for idx in trial_indices:
                    covered |= spheres_arr[idx]
                if not np.all(covered):
                    # Invalid move, revert
                    trial_indices = current_indices[:]
        
        # Reduce the trial code
        reduced_trial = reduce_code_with_counts(trial_indices)
        
        # Check if valid
        if reduced_trial:
             covered = np.zeros(N, dtype=bool)
             for idx in reduced_trial:
                 covered |= spheres_arr[idx]
             if np.all(covered):
                 if len(reduced_trial) < best_size:
                     best_size = len(reduced_trial)
                     best_code_indices = reduced_trial[:]
                 # Accept the move if it's better or equal (to explore)
                 if len(reduced_trial) <= len(current_indices):
                     current_indices = reduced_trial
                 else:
                     # With some probability, accept worse moves to escape local minima (Simulated Annealing style)
                     # But since we want to minimize, and time is limited, let's stick to hill climbing mostly
                     # or use a simple random restart if stuck?
                     # For now, accept if equal size to diversify.
                     if len(reduced_trial) == len(current_indices) and rng.random() < 0.5:
                         current_indices = reduced_trial

    # Final verification and conversion
    if not best_code_indices:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(all_words[i]) for i in best_code_indices]
