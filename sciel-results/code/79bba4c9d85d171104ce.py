"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # Safety margin before the 120s kill (using 115s to be safe)

    # ---- mapping helpers ----
    def word_to_int(w):
        val = 0
        for x in w:
            val = val * q + x
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w

    total = q ** n

    # ---- precompute radius balls for every word ----
    # For q=3, n=6, radius=1, ball size is 1 + 6*(3-1) = 13
    neighbor_lists = []
    for i in range(total):
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    ball.append(word_to_int(w))
            w[pos] = orig
        neighbor_lists.append(ball)

    # ---- small-n fallback (exhaustive greedy) ----
    if n <= 4:
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        code = []
        for w in all_words:
            if not uncovered:
                break
            if w in uncovered:
                code.append(list(w))
                for u in list(uncovered):
                    if sum(1 for i in range(n) if u[i] != w[i]) <= radius:
                        uncovered.remove(u)
        return code

    # ---- initial code: direct sum of the perfect ternary tetracode ----
    # Tetracode: length 4, radius 1, size 9
    # Tail: length 2, radius 0 (all words), size 9
    # Total: length 6, radius 1+0=1, size 81
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_tails = list(itertools.product(range(q), repeat=n - 4))
        code = [h + list(t) for h in tetracode for t in base_tails]
    else:
        # Fallback: greedy initial code
        uncovered = set(range(total))
        code = []
        for i in range(total):
            if not uncovered:
                break
            if i in uncovered:
                code.append(int_to_word(i))
                for nb in neighbor_lists[i]:
                    if nb in uncovered:
                        uncovered.discard(nb)

    code_indices = [word_to_int(c) for c in code]
    code_set = set(code_indices)

    # Coverage counts
    counts = np.zeros(total, dtype=np.int32)
    for ci in code_indices:
        for nb in neighbor_lists[ci]:
            counts[nb] += 1

    # Helper to check if a codeword can be removed
    def can_remove(ci):
        return np.all(counts[neighbor_lists[ci]] >= 2)

    # Helper to remove a codeword
    def remove_codeword(ci):
        idx = code_indices.index(ci)
        for nb in neighbor_lists[ci]:
            counts[nb] -= 1
        code_indices.pop(idx)
        code_set.discard(ci)

    # Helper to add a codeword
    def add_codeword(ci):
        code_indices.append(ci)
        code_set.add(ci)
        for nb in neighbor_lists[ci]:
            counts[nb] += 1

    # ---- Phase 1: Greedy Removal ----
    # Remove any redundant codewords
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 5.0:
                break
            ci = code_indices[i]
            if can_remove(ci):
                remove_codeword(ci)
                changed = True
                break  # Restart loop after change

    # ---- Phase 2: Simulated Annealing ----
    # We want to minimize size. 
    # Moves:
    # 1. Remove a random codeword. If valid, accept (size -1).
    # 2. If invalid, find the best replacement (max coverage of deficit) and swap (size unchanged).
    # 3. Occasionally, to escape local optima, we might accept a size-increasing move? 
    #    Actually, for set cover, "adding" without removing is rarely helpful unless we are in a bad state.
    #    But "Swapping" is the key.
    
    # To make it stronger, we implement a "2-exchange" or "Multi-swap" if simple swaps fail.
    # However, SA with swaps is usually sufficient.
    
    rng = random.Random(42)
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute for each word, which codewords cover it? 
    # Or just use counts.
    
    best_code_indices = list(code_indices)
    best_size = len(code_indices)
    
    while time.time() < deadline - 10.0:
        if T < T_min:
            # Reheat occasionally? Or just stop?
            # Let's try a few restarts if we get stuck.
            T = 0.1
            # Perturb: remove 2 random, add 2 random?
            if len(code_indices) > 2:
                # Simple perturbation
                idxs = rng.sample(range(len(code_indices)), 2)
                c1 = code_indices[idxs[0]]
                c2 = code_indices[idxs[1]]
                # Remove
                for c in [c1, c2]:
                    for nb in neighbor_lists[c]:
                        counts[nb] -= 1
                    code_indices.remove(c)
                    code_set.discard(c)
                # Add 2 random
                for _ in range(2):
                    new_c = rng.randint(0, total - 1)
                    if new_c not in code_set:
                        add_codeword(new_c)
            continue
            
        # Pick a random codeword to remove
        if not code_indices:
            break
            
        idx = rng.randint(0, len(code_indices) - 1)
        ci_out = code_indices[idx]
        
        # Temporarily remove
        for nb in neighbor_lists[ci_out]:
            counts[nb] -= 1
        code_indices.pop(idx)
        code_set.discard(ci_out)
        
        # Check coverage
        if np.all(counts >= 1):
            # Valid move, size decreased
            # Accept
            if len(code_indices) < best_size:
                best_size = len(code_indices)
                best_code_indices = list(code_indices)
            # Cooldown
            T *= cooling
            continue
        else:
            # Invalid move, need to fix
            # Find uncovered words
            uncovered = np.where(counts == 0)[0]
            
            # Find best replacement
            # We want a codeword c_in that maximizes the number of uncovered words it covers.
            # Score(c_in) = sum(1 for u in uncovered if u in ball(c_in))
            # Since uncovered is small (usually 1 or 2 words for radius 1), we can check candidates.
            # Candidates: All words in the space? 729.
            # For each candidate, check overlap with uncovered.
            
            best_score = -1
            best_c_in = -1
            
            # Optimization: Only check words that are in the ball of the uncovered words?
            # If a word u is uncovered, c_in must be in ball(u) to cover it.
            # So candidates are union of balls of uncovered words.
            candidate_set = set()
            for u in uncovered:
                candidate_set.update(neighbor_lists[u])
            
            for c_in in candidate_set:
                if c_in in code_set:
                    continue
                # Calculate score: how many uncovered words does c_in cover?
                score = 0
                for u in uncovered:
                    if u in neighbor_lists[c_in]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_c_in = c_in
            
            if best_c_in != -1 and best_score > 0:
                # Add the best replacement
                add_codeword(best_c_in)
                # Size is unchanged (removed 1, added 1)
                # This is a neutral move in terms of size, but changes the structure.
                # Accept with probability 1? Or based on T?
                # Since it restores validity, we accept it to continue the search.
                T *= cooling
            else:
                # Could not fix with a single swap. 
                # Revert the removal.
                add_codeword(ci_out)
                T *= cooling
                # If we can't fix, we are stuck.
                # Try a 2-swap? Or just reheat.
                if rng.random() < 0.1:
                    T = max(T, 0.5)

    # ---- Phase 3: Final Greedy Removal ----
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 2.0:
                break
            ci = code_indices[i]
            if can_remove(ci):
                remove_codeword(ci)
                changed = True
                break

    # Use the best found code if it's smaller
    if len(best_code_indices) < len(code_indices):
        code_indices = best_code_indices

    return [int_to_word(ci) for ci in code_indices]
# EVOLVE-BLOCK-END
