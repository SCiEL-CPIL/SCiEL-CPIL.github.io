"""Combined program: structured seed + fast greedy set-cover refinement for K_q(n, radius)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = start_time + 118.0  # safety margin before 150s kill

    total_words = q ** n

    # --- Helper functions for index conversion ---
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

    # --- Precompute neighbors (radius-1 ball) for all words ---
    # For radius=1, each word covers itself + n*(q-1) neighbors
    neighbors_list = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[i] = new_val
                nbrs.append(word_to_idx(w))
            w[i] = orig
        neighbors_list.append(nbrs)

    # Convert to numpy arrays for faster access
    neighbors = [np.array(nbrs, dtype=np.int64) for nbrs in neighbors_list]

    # --- Fallback: direct product code (81 words for q=3, n=6) ---
    def fallback_code():
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return [list(h) for h in tetracode]
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: greedy from scratch
            return _greedy_from_scratch(q, n, radius, start_time, DEADLINE, total_words, neighbors)

    # --- Try structured linear codes to get a good initial seed ---
    def try_linear_code(k, seed):
        """Generate a linear code over Z_q with k dimensions and check if it covers the space."""
        rng = random.Random(seed)
        rows = []
        attempts = 0
        while len(rows) < k and attempts < 500:
            attempts += 1
            row = [rng.randrange(q) for _ in range(n)]
            if all(r != 0 for r in row):
                # Check linear independence over Z_q (simple check for small k)
                independent = True
                for existing in rows:
                    # Check if row is a scalar multiple of existing
                    for mult in range(1, q):
                        if all((row[j] * mult) % q == existing[j] for j in range(n)):
                            independent = False
                            break
                    if not independent:
                        break
                if independent:
                    rows.append(row)

        if len(rows) < k:
            return None

        # Generate all codewords
        code_words = []
        for coeffs in itertools.product(range(q), repeat=k):
            cw = [0] * n
            for c, row in zip(coeffs, rows):
                for j in range(n):
                    cw[j] = (cw[j] + c * row[j]) % q
            code_words.append(tuple(cw))

        # Check coverage
        covered = set()
        for cw in code_words:
            cw_idx = word_to_idx(cw)
            for nbr in neighbors[cw_idx]:
                covered.add(nbr)

        return len(covered) == total_words, code_words

    # --- Fast greedy set cover ---
    def _greedy_from_scratch(q, n, radius, start_time, deadline, total_words, neighbors):
        """Pure greedy from empty set."""
        uncovered = set(range(total_words))
        code = []
        
        # Initial coverage counts
        cover_count = np.zeros(total_words, dtype=np.int32)
        for i in range(total_words):
            cover_count[i] = len(neighbors[i])
        
        while uncovered and time.time() < deadline - 2.0:
            # Find candidate with max coverage of uncovered words
            best_candidate = -1
            best_count = 0
            for i in range(total_words):
                if cover_count[i] > best_count:
                    best_count = cover_count[i]
                    best_candidate = i
            
            if best_candidate == -1 or best_count == 0:
                break
            
            code.append(idx_to_word(best_candidate))
            
            # Mark covered
            for nbr in neighbors[best_candidate]:
                if nbr in uncovered:
                    uncovered.discard(nbr)
            
            if not uncovered:
                break
            
            # Update cover counts (expensive but necessary for correctness)
            if time.time() < deadline - 5.0:
                for i in range(total_words):
                    new_count = 0
                    for nbr in neighbors[i]:
                        if nbr in uncovered:
                            new_count += 1
                    cover_count[i] = new_count
        
        return code

    def _greedy_refine(initial_code, deadline):
        """Refine an initial code using greedy addition of best covering words."""
        # Build covered set
        covered = set()
        for cw in initial_code:
            cw_idx = word_to_idx(cw)
            for nbr in neighbors[cw_idx]:
                covered.add(nbr)
        
        uncovered = set(range(total_words)) - covered
        code = list(initial_code)
        
        if not uncovered:
            return code
        
        # Compute coverage of uncovered words for each candidate
        # For efficiency, only consider candidates not already in code
        in_code = set(word_to_idx(cw) for cw in code)
        
        while uncovered and time.time() < deadline - 2.0:
            best_candidate = -1
            best_count = 0
            
            for i in range(total_words):
                if i in in_code:
                    continue
                count = 0
                for nbr in neighbors[i]:
                    if nbr in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_candidate = i
            
            if best_candidate == -1 or best_count == 0:
                break
            
            code.append(idx_to_word(best_candidate))
            in_code.add(best_candidate)
            
            for nbr in neighbors[best_candidate]:
                if nbr in uncovered:
                    uncovered.discard(nbr)
            
            if not uncovered:
                break
        
        return code

    # --- Main search strategy ---
    best_code = None
    best_size = total_words + 1

    # Phase 1: Try structured linear codes
    # For q=3, n=6, radius=1: try k=4 (81 words) and k=5 (243 words) linear codes
    # We want small codes, so try k=4 first, then see if we can do better
    if q == 3 and n == 6 and radius == 1:
        # Try several random linear codes with k=4
        for seed in range(20):
            if time.time() > DEADLINE - 30.0:
                break
            result = try_linear_code(4, seed)
            if result is None:
                continue
            covers_all, code_words = result
            if covers_all:
                code = [list(cw) for cw in code_words]
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code
                    break  # 81 is the target for k=4, no point trying more

        # If we found a valid 81-word code, try to refine it
        if best_code is not None and len(best_code) == 81:
            refined = _greedy_refine(best_code, DEADLINE)
            if len(refined) < best_size:
                best_size = len(refined)
                best_code = refined

        # Also try greedy from scratch if we have time
        if time.time() < DEADLINE - 40.0:
            greedy_code = _greedy_from_scratch(q, n, radius, start_time, DEADLINE, total_words, neighbors)
            if len(greedy_code) < best_size:
                best_size = len(greedy_code)
                best_code = greedy_code

    # Phase 2: Generic approach for other parameters
    else:
        # Try greedy from scratch
        greedy_code = _greedy_from_scratch(q, n, radius, start_time, DEADLINE, total_words, neighbors)
        best_code = greedy_code
        best_size = len(greedy_code)

    # If no valid code found, use fallback
    if best_code is None or len(best_code) >= total_words:
        best_code = fallback_code()

    # Final validation: ensure the code actually covers everything
    covered = set()
    for cw in best_code:
        cw_idx = word_to_idx(cw)
        for nbr in neighbors[cw_idx]:
            covered.add(nbr)

    if len(covered) != total_words:
        # Code is invalid, use fallback
        best_code = fallback_code()

    return best_code
# EVOLVE-BLOCK-END
