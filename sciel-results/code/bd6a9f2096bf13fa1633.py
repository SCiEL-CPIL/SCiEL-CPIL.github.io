"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius r.
    Uses a greedy initialization followed by simulated annealing to minimize the number of codewords.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple random or greedy approach
        return _fallback_construct(q, n, radius)

    # Specific optimization for q=3, n=8, r=1
    # Total space size: 3^8 = 6561
    # Sphere size: 1 + n*(q-1) = 1 + 8*2 = 17
    # Lower bound: ceil(6561 / 17) = 386.
    # Best known: 486.
    
    # Precompute mapping from word to index
    # word is a tuple of 8 ints in 0..2
    # index is 0..6560
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)

    # Precompute neighbors for all 6561 words
    # neighbor[i] is a list of indices at Hamming distance 1 from i
    # Also include self? No, sphere includes self. Let's precompute the full sphere.
    spheres = np.zeros((q**n, n*(q-1)+1), dtype=np.int32)
    sphere_size = 1 + n*(q-1)
    
    for idx in range(q**n):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors.append(word_to_idx(tuple(new_w)))
        spheres[idx] = neighbors

    # Greedy Initialization
    # Start with a large set of codewords, then prune.
    # Or start with random set and add/remove.
    
    # Let's start with a random set of size ~600 and refine.
    # Or better: start with the direct product construction (9 * 3^4 = 729 words) and prune?
    # The seed program returns 729 words. We want to go lower.
    
    # Strategy: 
    # 1. Generate a random set of codewords of size M (e.g., 600).
    # 2. Check coverage. If not covered, add random words until covered.
    # 3. Simulated Annealing:
    #    - State: set of codewords (represented as a boolean array of size 6561 or a set of indices).
    #    - Move: 
    #       a. Remove a random codeword if coverage is maintained.
    #       b. Move a random codeword to a new random position.
    #       c. Add a random codeword (if we allow size increase to escape local minima).
    #    - Energy: Number of codewords + penalty for uncovered points.
    
    # For speed, we need efficient coverage tracking.
    # Let's use a NumPy array `covered_count` of size 6561.
    # covered_count[i] = number of codewords that cover point i.
    
    def add_codeword(idx, covered_count):
        for nb in spheres[idx]:
            covered_count[nb] += 1
    
    def remove_codeword(idx, covered_count):
        for nb in spheres[idx]:
            covered_count[nb] -= 1
    
    def is_covered(covered_count):
        return np.all(covered_count > 0)
    
    def uncovered_count(covered_count):
        return np.sum(covered_count == 0)

    # Initialize
    random.seed(42)
    
    # Start with a set of size 500
    current_size = 500
    codewords = set(random.sample(range(q**n), current_size))
    covered_count = np.zeros(q**n, dtype=np.int32)
    for c in codewords:
        add_codeword(c, covered_count)
    
    # Ensure coverage by adding random points if needed
    while not is_covered(covered_count):
        # Find an uncovered point
        uncovered_idx = np.where(covered_count == 0)[0]
        if len(uncovered_idx) == 0:
            break
        # Pick a random point near an uncovered one? Or just random.
        # Let's pick a random point from the space.
        new_c = random.randint(0, q**n - 1)
        if new_c not in codewords:
            codewords.add(new_c)
            add_codeword(new_c, covered_count)
            current_size += 1
        else:
            # If we keep picking existing, just force add a neighbor of an uncovered point
            target = uncovered_idx[random.randint(0, len(uncovered_idx)-1)]
            # Find a point that covers target
            # The point itself covers target
            new_c = target
            if new_c not in codewords:
                codewords.add(new_c)
                add_codeword(new_c, covered_count)
                current_size += 1
            else:
                # Should not happen if logic is correct, but safety break
                break

    best_size = current_size
    best_codewords = set(codewords)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    steps = 0
    
    while T > min_T and time.time() < deadline:
        steps += 1
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to remove a codeword
            if len(codewords) > 100:
                c_to_remove = random.choice(list(codewords))
                remove_codeword(c_to_remove, covered_count)
                new_uncovered = uncovered_count(covered_count)
                if new_uncovered == 0:
                    # Successful removal
                    codewords.discard(c_to_remove)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_codewords = set(codewords)
                else:
                    # Revert
                    add_codeword(c_to_remove, covered_count)
        elif move_type < 0.7:
            # Try to move a codeword
            c_old = random.choice(list(codewords))
            c_new = random.randint(0, q**n - 1)
            if c_new != c_old and c_new not in codewords:
                remove_codeword(c_old, covered_count)
                add_codeword(c_new, covered_count)
                new_uncovered = uncovered_count(covered_count)
                # Energy change: size is same, but coverage might change
                # We accept if coverage is maintained, or with probability exp(-delta/T) if worse
                # Here, "worse" means uncovered points increased.
                old_uncovered = 0 # We assume we are always in a covered state or we track it?
                # Actually, we should track the state of coverage.
                # If we are in a valid state, old_uncovered is 0.
                # If new_uncovered > 0, we have invalid state.
                
                if new_uncovered == 0:
                    # Valid state
                    codewords.discard(c_old)
                    codewords.add(c_new)
                    # Size unchanged
                else:
                    # Invalid state, accept with probability
                    prob = np.exp(-new_uncovered * 0.1 / T)
                    if random.random() < prob:
                        codewords.discard(c_old)
                        codewords.add(c_new)
                    else:
                        # Revert
                        remove_codeword(c_new, covered_count)
                        add_codeword(c_old, covered_count)
            elif c_new == c_old:
                pass
            else:
                # c_new already in codewords, swap?
                # Just ignore for simplicity
                pass
        else:
            # Try to add a codeword (to escape local minima where removals are stuck)
            # Only if we are in a valid state? Or always?
            # Let's only add if we are in a valid state, to explore larger spaces
            if uncovered_count(covered_count) == 0:
                c_new = random.randint(0, q**n - 1)
                if c_new not in codewords:
                    add_codeword(c_new, covered_count)
                    codewords.add(c_new)
                    current_size += 1
                    # Accept addition with probability based on T?
                    # Usually we want to minimize size, so additions are "bad" but help exploration.
                    # Let's always accept additions in SA if we are in a good state, 
                    # hoping that subsequent removals will find a better configuration.
                    # Or use probability exp(1/T) which is > 1, so always accept?
                    # Let's just accept.
    
    # Final cleanup: try to remove any redundant codewords greedily
    # Sort by something? Just iterate multiple times.
    for _ in range(10):
        removed_any = False
        for c in list(codewords):
            remove_codeword(c, covered_count)
            if uncovered_count(covered_count) == 0:
                codewords.discard(c)
                current_size -= 1
                removed_any = True
            else:
                add_codeword(c, covered_count)
        if not removed_any:
            break

    # Return the best found
    final_codewords = best_codewords
    # Convert set of indices to list of lists
    result = [list(idx_to_word(idx)) for idx in final_codewords]
    return result

def _fallback_construct(q, n, radius):
    # Simple random construction for non-standard parameters
    random.seed(42)
    codewords = set()
    # Add random words until we have a decent number
    # This is a dummy implementation, task is specific to q=3, n=8
    for _ in range(100):
        w = tuple(random.randint(0, q-1) for _ in range(n))
        codewords.add(w)
    return [list(w) for w in codewords]

# EVOLVE-BLOCK-END
