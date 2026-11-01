import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    if q != 3 or n != 6 or radius != 1:
        return _greedy_set_cover(q, n, radius, deadline)

    space_size = q ** n  # 729

    # Precompute index <-> word conversions
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors (distance <= radius) for every word in the space
    # neighbors[i] is a set of indices at distance <= 1 from i
    neighbors = []
    for i in range(space_size):
        w = idx_to_word(i)
        nbs = {i}  # include self
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    nbs.add(word_to_idx(w))
            w[pos] = orig
        neighbors.append(nbs)

    # Also build a list version for iteration
    neighbors_list = [list(nb) for nb in neighbors]

    # Tetracode direct-sum initial construction: 9 * 3^(6-4) = 81 words
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    initial_codewords = [h + list(tail) for h in tetracode for tail in tails]
    
    # Convert to indices
    initial_indices = [word_to_idx(w) for w in initial_codewords]

    def build_counts(indices):
        """Build coverage count array from a list of codeword indices."""
        counts = np.zeros(space_size, dtype=np.int32)
        for ci in indices:
            counts[neighbors_list[ci]] += 1
        return counts

    def greedy_remove(cw_indices, counts, deadline):
        """Iteratively remove redundant codewords. Returns (list, counts)."""
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    break
                ci = current[i]
                can_remove = True
                for nb in neighbors_list[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors_list[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    changed = True
        return current, cnt

    def local_search_swap(cw_indices, counts, deadline, max_iterations=50000):
        """
        Local search: try swaps to reduce size.
        We iterate over codewords and try to replace them with a random other word
        if it allows for a net reduction or enables further removals.
        """
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        
        rng = random.Random(42)
        improved = True
        iterations = 0
        
        while improved and time.time() < deadline and iterations < max_iterations:
            improved = False
            iterations += 1
            # Shuffle to break symmetry and avoid local minima
            if iterations % 100 == 0:
                rng.shuffle(current)
                
            # Phase 1: Greedy Removal
            removal_made = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    return current, cnt
                ci = current[i]
                can_remove = True
                for nb in neighbors_list[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors_list[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    improved = True
                    removal_made = True
                    break # Restart scan after removal
            
            if removal_made:
                continue

            # Phase 2: Try Swaps
            if not current:
                break
                
            idx_to_remove = rng.randint(0, len(current) - 1)
            ci = current[idx_to_remove]
            
            # Find a candidate to add
            essential = [nb for nb in neighbors_list[ci] if cnt[nb] == 1]
            
            if not essential:
                for nb in neighbors_list[ci]:
                    cnt[nb] -= 1
                current.pop(idx_to_remove)
                current_set.discard(ci)
                improved = True
                continue

            found_swap = False
            for _ in range(20):
                if time.time() > deadline:
                    return current, cnt
                cand = rng.randint(0, space_size - 1)
                if cand in current_set:
                    continue
                
                cand_nbs = neighbors_list[cand]
                cand_set = neighbors[cand]
                
                if not all(e in cand_set for e in essential):
                    continue
                
                valid = True
                for nb in neighbors_list[ci]:
                    if cnt[nb] == 1 and nb not in cand_set:
                        valid = False
                        break
                
                if valid:
                    for nb in neighbors_list[ci]:
                        cnt[nb] -= 1
                    for nb in cand_nbs:
                        cnt[nb] += 1
                    
                    current[idx_to_remove] = cand
                    current_set.discard(ci)
                    current_set.add(cand)
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap and not improved:
                break

        return current, cnt

    # Main search loop
    best_size = float('inf')
    best_code = None
    
    # 1. Initial greedy reduction
    counts = build_counts(initial_indices)
    code, counts = greedy_remove(initial_indices, counts, deadline - 60)
    
    # 2. Local search with swaps
    code, counts = local_search_swap(code, counts, deadline - 20, max_iterations=100000)
    
    if len(code) < best_size:
        best_size = len(code)
        best_code = code.copy()

    # 3. Randomized restarts with different initial orderings or perturbations
    if time.time() < deadline - 10 and best_size > 73:
        current_best = best_code.copy()
        current_counts = build_counts(current_best)
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            perturbed = current_best.copy()
            perturbed_counts = current_counts.copy()
            
            rng = random.Random(restart)
            for _ in range(5):
                if len(perturbed) < 2:
                    break
                i = rng.randint(0, len(perturbed) - 1)
                ci = perturbed[i]
                for _ in range(10):
                    cand = rng.randint(0, space_size - 1)
                    if cand in set(perturbed):
                        continue
                    cand_set = neighbors[cand]
                    essential = [nb for nb in neighbors_list[ci] if perturbed_counts[nb] == 1]
                    if not all(e in cand_set for e in essential):
                        continue
                    valid = True
                    for nb in neighbors_list[ci]:
                        if perturbed_counts[nb] == 1 and nb not in cand_set:
                            valid = False
                            break
                    if valid:
                        for nb in neighbors_list[ci]:
                            perturbed_counts[nb] -= 1
                        for nb in neighbors_list[cand]:
                            perturbed_counts[nb] += 1
                        perturbed[i] = cand
                        break
            
            improved_code, improved_counts = local_search_swap(perturbed, perturbed_counts, deadline - 5, max_iterations=20000)
            
            if len(improved_code) < best_size:
                best_size = len(improved_code)
                best_code = improved_code.copy()
                current_best = best_code.copy()
                current_counts = build_counts(current_best)

    # Final polish
    if best_code is not None and time.time() < deadline - 2:
        final_counts = build_counts(best_code)
        final_code, final_counts = greedy_remove(best_code, final_counts, deadline - 1)
        final_code, final_counts = local_search_swap(final_code, final_counts, deadline, max_iterations=5000)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)

    result = [idx_to_word(ci) for ci in best_code]
    return result

def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    coverage = {}
    for i, w in enumerate(all_words):
        covered = set()
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                covered.add(j)
        coverage[i] = covered

    uncovered = set(range(total))
    code = []

    while uncovered and time.time() < deadline - 5:
        best_i = -1
        best_count = 0
        for i in range(total):
            count = len(uncovered & coverage[i])
            if count > best_count:
                best_count = count
                best_i = i
            if time.time() > deadline - 5:
                break
        if best_i == -1:
            break
        code.append(list(all_words[best_i]))
        uncovered -= coverage[best_i]

    return code
