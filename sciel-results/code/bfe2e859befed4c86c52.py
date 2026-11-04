import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors as bitsets using numpy uint64
    # N = 729, so we need 729/64 = 12 uint64s per word
    num_words = N
    bits_per_word = 64
    num_blocks = (num_words + bits_per_word - 1) // bits_per_word
    # neighbor_bits[i] is an array of num_blocks uint64s
    neighbor_bits = np.zeros((num_words, num_blocks), dtype=np.uint64)
    # Also store the list of neighbors for quick iteration
    neighbors = []
    
    for i, c in enumerate(words):
        neigh = [i]
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                idx = word_to_idx[tuple(new_w)]
                neigh.append(idx)
        neighbors.append(neigh)
        
        # Set bits in neighbor_bits
        for nb in neigh:
            block_idx = nb // bits_per_word
            bit_idx = nb % bits_per_word
            neighbor_bits[i, block_idx] |= (1 << bit_idx)

    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    selected = []
    
    # Precompute coverage counts for greedy? No, we compute on the fly.
    # But we can use the bitsets to speed up counting?
    # Actually, for N=729, simple loops are fast enough.
    
    while uncovered.any():
        best_idx = -1
        best_count = 0
        for j in range(N):
            # Count uncovered neighbors
            # Using numpy: sum(uncovered[neighbors[j]])
            # But neighbors[j] is a list.
            # Let's stick to the list for simplicity, it's fast enough.
            count = 0
            for nb in neighbors[j]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = j
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered[nb] = False
            
        if time.time() > deadline:
            break
    
    # Initial solution size: len(selected)
    # Now, try to reduce the size using local search.
    
    # Maintain coverage counts
    coverage_count = np.zeros(N, dtype=np.int32)
    for j in selected:
        for nb in neighbors[j]:
            coverage_count[nb] += 1
            
    # Helper function to check if a codeword is redundant
    def is_redundant(j, cov):
        for nb in neighbors[j]:
            if cov[nb] == 1:
                return False
        return True

    # Helper function to try to reduce a set
    def try_reduce(selected_set, cov, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(selected_set)):
                j = selected_set[idx]
                if is_redundant(j, cov):
                    for nb in neighbors[j]:
                        cov[nb] -= 1
                    selected_set.pop(idx)
                    changed = True
                    break
        return selected_set

    # Local Search: Add-Remove Strategy
    # We will try to add a random codeword, then reduce. If size decreases, keep it.
    # We will also try a deterministic version: iterate through all possible additions.
    
    best_selected = list(selected)
    best_cov = coverage_count.copy()
    
    # First, try to reduce the initial greedy solution
    best_selected = try_reduce(best_selected, best_cov, deadline)
    
    # Now, try improvements
    # We use a random seed for reproducibility
    random.seed(42)
    
    # To make the search more effective, we can prioritize adding codewords that are "far" from the current set
    # or simply random.
    
    # Let's try a loop of random restarts of the local search
    # Actually, the "add then reduce" is a form of local search.
    
    # We can also try "swap" moves: remove one, add one.
    # But "add then reduce" is simpler and often effective.
    
    # To speed up, we can limit the number of iterations.
    max_iters = 1000
    iter_count = 0
    
    while time.time() < deadline - 2.0 and iter_count < max_iters:
        iter_count += 1
        # Pick a random candidate to add
        if len(best_selected) < N:
            # To avoid adding a word that is already in the set
            # We can just pick a random index and check
            add_idx = random.randint(0, N - 1)
            
            # If it's already in the set, skip (or we could handle it, but skipping is easier)
            if add_idx in best_selected:
                continue
            
            # Add it
            new_selected = list(best_selected)
            new_selected.append(add_idx)
            
            # Create new coverage counts
            new_cov = best_cov.copy()
            for nb in neighbors[add_idx]:
                new_cov[nb] += 1
            
            # Try to reduce
            new_selected = try_reduce(new_selected, new_cov, deadline)
            
            if len(new_selected) < len(best_selected):
                best_selected = new_selected
                best_cov = new_cov
                # If we improved, we can restart the random search or continue
                # Restarting might help escape local optima, but let's just continue
                # for now.
            else:
                # No improvement, discard
                pass
        
        if time.time() > deadline:
            break
    
    # Final check: ensure the selected set is valid
    # (It should be, by construction)
    
    # Convert indices back to words
    codewords = [words[i] for i in best_selected]
    
    return codewords
