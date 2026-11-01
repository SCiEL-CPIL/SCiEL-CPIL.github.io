import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    if q != 3 or n != 6 or radius != 1:
        return _greedy_set_cover(q, n, radius, deadline)

    space_size = q ** n  # 729

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

    # Precompute neighbors
    neighbors = []
    neighbor_sets = []
    for i in range(space_size):
        w = idx_to_word(i)
        nbs = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    nbs.append(word_to_idx(w))
            w[pos] = orig
        neighbors.append(nbs)
        neighbor_sets.append(set(nbs))

    # Initial construction: Tetracode direct sum
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    initial_codewords = [h + list(tail) for h in tetracode for tail in tails]
    initial_indices = [word_to_idx(w) for w in initial_codewords]

    def build_counts(cw_indices):
        counts = np.zeros(space_size, dtype=np.int32)
        for ci in cw_indices:
            counts[neighbors[ci]] += 1
        return counts

    def is_redundant(ci, counts):
        for nb in neighbors[ci]:
            if counts[nb] == 1:
                return False
        return True

    def greedy_remove(cw_indices, counts):
        current = list(cw_indices)
        cnt = counts.copy()
        changed = True
        while changed:
            changed = False
            for i in range(len(current) - 1, -1, -1):
                if is_redundant(current[i], cnt):
                    for nb in neighbors[current[i]]:
                        cnt[nb] -= 1
                    current.pop(i)
                    changed = True
        return current, cnt

    def local_search(cw_indices, counts, max_iters=1000):
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Phase 1: Remove redundant
            removed = False
            for i in range(len(current) - 1, -1, -1):
                if is_redundant(current[i], cnt):
                    for nb in neighbors[current[i]]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(current[i])
                    removed = True
            if removed:
                continue
            
            # Phase 2: Try swaps to enable removal
            # Pick a random codeword to try moving
            # We try to replace c with c' such that c' covers essential neighbors of c
            # and hopefully makes another codeword redundant.
            
            # Find a codeword with "loose" coverage?
            # Just try random swaps.
            swapped = False
            if current:
                idx_to_move = random.randint(0, len(current) - 1)
                c = current[idx_to_move]
                essential = [nb for nb in neighbors[c] if cnt[nb] == 1]
                
                # If no essential neighbors, we can remove it (should have been caught in Phase 1)
                if not essential:
                    for nb in neighbors[c]:
                        cnt[nb] -= 1
                    current.pop(idx_to_move)
                    current_set.discard(c)
                    continue

                # Try a few random candidates
                for _ in range(20):
                    cand = random.randint(0, space_size - 1)
                    if cand in current_set:
                        continue
                    
                    # Check if cand covers all essential neighbors of c
                    if not all(e in neighbor_sets[cand] for e in essential):
                        continue
                    
                    # Check if swap is valid (no new uncovered words)
                    # Words in neighbors[c] \ neighbors[cand] must have count > 1
                    valid = True
                    for nb in neighbors[c]:
                        if cnt[nb] == 1 and nb not in neighbor_sets[cand]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Perform swap
                    for nb in neighbors[c]:
                        cnt[nb] -= 1
                    for nb in neighbors[cand]:
                        cnt[nb] += 1
                    current[idx_to_move] = cand
                    current_set.discard(c)
                    current_set.add(cand)
                    swapped = True
                    break
            
            if not swapped:
                break
                
        return current, cnt

    def repair(cw_indices, counts):
        """Add codewords to make the code valid again."""
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        
        # Find uncovered words
        uncovered = [i for i in range(space_size) if cnt[i] == 0]
        
        while uncovered:
            if time.time() > deadline - 2:
                break
            # Pick a random uncovered word
            u = random.choice(uncovered)
            # Pick a random neighbor of u to add
            nbs = neighbors[u]
            # Prefer neighbors that cover the most uncovered words?
            # For simplicity, pick random neighbor
            c = random.choice(nbs)
            if c in current_set:
                # If already in code, it means cnt[c] should be > 0?
                # No, if c is in code, it covers itself. If u is uncovered, c != u.
                # But c might cover u. If c is in code, u should be covered.
                # So c is not in code.
                pass
            
            # Add c
            current.append(c)
            current_set.add(c)
            for nb in neighbors[c]:
                cnt[nb] += 1
                if cnt[nb] == 1 and nb in uncovered:
                    # It's covered now
                    # We can remove it from uncovered list, but list ops are slow.
                    # Just rebuild uncovered list occasionally or check at end.
                    pass
        
        # Rebuild uncovered to clean up
        # This is O(N), which is fine.
        uncovered = [i for i in range(space_size) if cnt[i] == 0]
        return current, cnt, uncovered

    # Main ILS Loop
    best_code = None
    best_size = float('inf')
    
    # Initial Greedy Remove
    init_counts = build_counts(initial_indices)
    curr_code, curr_counts = greedy_remove(initial_indices, init_counts)
    
    # Local Search on initial
    curr_code, curr_counts = local_search(curr_code, curr_counts)
    
    if len(curr_code) < best_size:
        best_size = len(curr_code)
        best_code = curr_code.copy()
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        # Perturb: Remove 1 random codeword
        if curr_code:
            idx_to_remove = rng.randint(0, len(curr_code) - 1)
            c_to_remove = curr_code[idx_to_remove]
            
            # Create a new state by removing c_to_remove
            perturbed_code = curr_code.copy()
            perturbed_code.pop(idx_to_remove)
            
            # Rebuild counts for perturbed code?
            # Or just modify curr_counts?
            # Modifying is faster.
            # But we need to restore if we fail?
            # ILS usually works on a copy or commits the perturbation.
            # Let's commit the perturbation: we now have an invalid code.
            
            # Update counts
            for nb in neighbors[c_to_remove]:
                curr_counts[nb] -= 1
            
            # Repair
            curr_code, curr_counts, uncovered = repair(perturbed_code, curr_counts)
            
            # If repair added too many, try to remove redundant
            curr_code, curr_counts = greedy_remove(curr_code, curr_counts)
            
            # Local Search
            curr_code, curr_counts = local_search(curr_code, curr_counts)
            
            if len(curr_code) < best_size:
                best_size = len(curr_code)
                best_code = curr_code.copy()
                
            # If we hit the lower bound, stop
            if best_size <= 71:
                break
        else:
            break
            
    # Final Polish
    if best_code is not None:
        final_counts = build_counts(best_code)
        best_code, final_counts = local_search(best_code, final_counts, max_iters=5000)
        if len(best_code) < best_size:
            best_size = len(best_code)

    return [idx_to_word(ci) for ci in best_code]

def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # Precompute coverage sets
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
