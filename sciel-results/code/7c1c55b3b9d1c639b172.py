"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17
        
        # Precompute all spheres using bitmask representation
        # Each word in {0,1,2}^8 maps to an index 0..6560
        # Precompute for each index, which indices are within distance 1
        def word_to_int(w):
            """Convert a tuple of digits to an integer (base-3 encoding)."""
            result = 0
            for d in w:
                result = result * 3 + d
            return result
        
        def int_to_word(idx):
            """Convert an integer back to a tuple of digits (base-3)."""
            w = []
            for _ in range(n):
                w.append(idx % 3)
                idx //= 3
            return tuple(reversed(w))
        
        # Precompute sphere masks: for each word index, the list of indices it covers
        # This is a list of 6561 lists, each with 17 elements
        spheres = []
        for idx in range(total_space):
            w = int_to_word(idx)
            covered = [idx]
            for pos in range(n):
                for d in range(q):
                    if d != w[pos]:
                        new_w = list(w)
                        new_w[pos] = d
                        covered.append(word_to_int(tuple(new_w)))
            spheres.append(covered)
        
        # Greedy set cover: repeatedly pick the codeword that covers the most uncovered words
        uncovered = np.ones(total_space, dtype=bool)
        uncovered_count = total_space
        code_indices = []
        
        # For efficiency, we'll maintain an array of how many times each word is covered
        # by the selected codewords, and use a priority-based selection
        
        # Precompute: for each candidate word, its sphere as a numpy array
        sphere_arrays = [np.array(s, dtype=np.int32) for s in spheres]
        
        # Greedy selection
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Find the word that covers the most uncovered words
            best_idx = -1
            best_count = -1
            
            # Sample-based approach for speed: check all words but use vectorized ops
            # For 6561 words, this is feasible
            for idx in range(total_space):
                if not uncovered_count > 0:
                    break
                # Count how many of this word's sphere are still uncovered
                sphere_arr = sphere_arrays[idx]
                count = np.sum(uncovered[sphere_arr])
                if count > best_count:
                    best_count = count
                    best_idx = idx
                    if best_count == sphere_size:
                        break  # Can't do better
            
            if best_idx == -1:
                break
            
            code_indices.append(best_idx)
            # Mark covered words
            sphere_arr = sphere_arrays[best_idx]
            uncovered[sphere_arr] = False
            uncovered_count = int(np.sum(uncovered))
        
        # Now we have a greedy code. Let's try to reduce it via local search.
        # Convert to list of words
        code = [int_to_word(idx) for idx in code_indices]
        
        # Local search: try to remove redundant codewords
        # Use cover_count approach
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[spheres[idx]] += 1
        
        # Try multiple passes of removal
        m = len(code_indices)
        active = [True] * m
        
        for pass_num in range(20):
            if time.time() > deadline:
                break
            
            # Shuffle order for diversity
            order = list(range(m))
            random.shuffle(order)
            
            any_removed = False
            for i in order:
                if not active[i]:
                    continue
                
                # Check if codeword i is redundant
                redundant = True
                for covered_idx in spheres[code_indices[i]]:
                    if cover_count[covered_idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    active[i] = False
                    any_removed = True
                    for covered_idx in spheres[code_indices[i]]:
                        cover_count[covered_idx] -= 1
            
            if not any_removed:
                break
        
        final_code = [int_to_word(code_indices[i]) for i in range(m) if active[i]]
        
        # Verify coverage
        if len(final_code) < total_space:
            verify_cover = np.zeros(total_space, dtype=bool)
            for idx in [code_indices[i] for i in range(m) if active[i]]:
                verify_cover[spheres[idx]] = True
            if np.all(verify_cover):
                return [list(w) for w in final_code]
        
        # Fallback: return greedy code
        return [list(w) for w in code]
    
    else:
        # General case: use the tetracode-based construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback: return all words (valid but not minimal)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
