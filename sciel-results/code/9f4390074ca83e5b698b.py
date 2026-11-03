import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        if n <= 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n  # 729
    
    # --- 1. Precompute Data Structures ---
    
    # Helper: Convert index to word tuple
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[n - 1 - i] = temp % 3
            temp //= 3
        return tuple(w)
    
    # Helper: Convert word tuple to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx

    # Precompute spheres (Hamming balls of radius 1)
    # spheres[i] is a list of indices
    spheres = [None] * total_words
    sphere_sets = [None] * total_words  # For O(1) membership testing
    for idx in range(total_words):
        w = list(idx_to_word(idx))
        sphere_set = {idx}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                sphere_set.add(word_to_idx(w))
                w[pos] = orig_val
        spheres[idx] = list(sphere_set)
        sphere_sets[idx] = sphere_set
    
    # Precompute coverers: for each word, which codewords cover it
    # coverers[w] = list of codeword indices c such that w in spheres[c]
    coverers = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in spheres[c]:
            coverers[w].append(c)
    
    # --- 2. Initial Code Construction ---
    # Direct sum of tetracode (9 words, len 4) and all words on remaining n-4 coords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    initial_indices = [word_to_idx(cw) for cw in initial_codewords]
    
    # --- 3. Helper Functions for Search ---
    
    def compute_counts(code_indices):
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            counts[spheres[idx]] += 1
        return counts

    def reduce_code(code_indices, counts=None):
        """
        Greedily remove redundant codewords.
        Returns (new_code_indices, new_counts)
        """
        if counts is None:
            counts = compute_counts(code_indices)
        
        code_set = set(code_indices)
        changed = True
        while changed:
            changed = False
            for idx in list(code_set):
                # Check if all neighbors are covered by >1 codeword
                # If min(counts[spheres[idx]]) > 1, then idx is redundant
                if np.min(counts[spheres[idx]]) > 1:
                    code_set.remove(idx)
                    counts[spheres[idx]] -= 1
                    changed = True
        
        return list(code_set), counts

    def is_valid_swap(c_remove, c_add, code_set, counts):
        """
        Check if replacing c_remove with c_add keeps the code valid.
        c_remove must be in code_set, c_add must not be in code_set.
        """
        if c_remove not in code_set or c_add in code_set:
            return False
        
        # Words that become uncovered if we remove c_remove
        # These are words in Sphere(c_remove) that have count == 1
        # We need Sphere(c_add) to cover all these words.
        
        sphere_remove = spheres[c_remove]
        sphere_add_set = sphere_sets[c_add]
        
        for w in sphere_remove:
            if counts[w] == 1:
                if w not in sphere_add_set:
                    return False
        return True

    # --- 4. Main Search Loop ---
    
    rng = random.Random(42)
    
    # Start with initial code
    current_code, current_counts = reduce_code(initial_indices)
    best_code = current_code[:]
    best_counts = current_counts.copy()
    best_len = len(best_code)
    
    all_indices = list(range(total_words))
    
    no_improve_count = 0
    max_no_improve = 30  # Reduced to allow more perturbations
    
    while time.time() < deadline:
        # 1. Reduce
        current_code, current_counts = reduce_code(current_code, current_counts)
        
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code[:]
            best_counts = current_counts.copy()
            no_improve_count = 0
            if best_len <= 71: # Theoretical lower bound is 71, best known 73. If we hit 71, we are done.
                break
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Perturb: try multiple random valid swaps
            if current_code:
                current_set = set(current_code)
                found_valid = False
                
                # Try up to 200 random swaps
                for _ in range(200):
                    c_remove = rng.choice(current_code)
                    # Use coverers to find candidates that might be valid
                    # Find a critical word in sphere of c_remove
                    critical = [w for w in spheres[c_remove] if current_counts[w] == 1]
                    if not critical:
                        continue
                    
                    # Pick a random critical word and get its coverers
                    crit_w = rng.choice(critical)
                    candidates = [c for c in coverers[crit_w] if c not in current_set]
                    if not candidates:
                        continue
                    
                    c_add = rng.choice(candidates)
                    
                    if is_valid_swap(c_remove, c_add, current_set, current_counts):
                        current_set.remove(c_remove)
                        current_set.add(c_add)
                        current_counts[spheres[c_remove]] -= 1
                        current_counts[spheres[c_add]] += 1
                        current_code = list(current_set)
                        found_valid = True
                        no_improve_count = 0
                        break
                
                if not found_valid:
                    # Reset to best code to avoid drifting too far
                    current_code = best_code[:]
                    current_counts = best_counts.copy()
                    no_improve_count = 0
            continue

        # 2. Try to find a beneficial swap
        improved = False
        current_set = set(current_code)
        
        # Randomize order of removal candidates
        candidates = current_code[:]
        rng.shuffle(candidates)
        
        for c_remove in candidates:
            if c_remove not in current_set:
                continue
            
            # Find U: words in Sphere(c_remove) with count 1
            critical_words = [w for w in spheres[c_remove] if current_counts[w] == 1]
            
            if not critical_words:
                continue

            # Get candidate c_add values from coverers of critical words
            # Intersect coverers of all critical words
            candidate_set = set(coverers[critical_words[0]])
            for w in critical_words[1:]:
                candidate_set &= set(coverers[w])
            
            # Remove codewords already in current code
            candidate_set -= current_set
            
            if not candidate_set:
                continue
            
            # Try candidates in random order
            candidate_list = list(candidate_set)
            rng.shuffle(candidate_list)
            
            for c_add in candidate_list:
                # Perform swap
                current_set.remove(c_remove)
                current_set.add(c_add)
                
                # Update counts
                current_counts[spheres[c_remove]] -= 1
                current_counts[spheres[c_add]] += 1
                
                # Try to reduce
                new_code, new_counts = reduce_code(list(current_set), current_counts)
                
                if len(new_code) < len(current_code):
                    # Accept
                    current_code = new_code
                    current_counts = new_counts
                    current_set = set(current_code)
                    improved = True
                    break
                else:
                    # Revert
                    current_set.remove(c_add)
                    current_set.add(c_remove)
                    current_counts[spheres[c_add]] -= 1
                    current_counts[spheres[c_remove]] += 1
            
            if improved:
                break
            if time.time() > deadline - 5.0:
                break

    # Final check: ensure best_code is valid and converted to lists
    # Verify coverage
    final_counts = np.zeros(total_words, dtype=np.int32)
    for idx in best_code:
        final_counts[spheres[idx]] += 1
    if np.any(final_counts == 0):
        # Fallback to initial if something went wrong
        return [list(idx_to_word(idx)) for idx in initial_indices]
    
    return [list(idx_to_word(idx)) for idx in best_code]
