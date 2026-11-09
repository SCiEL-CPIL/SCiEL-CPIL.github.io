import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code for q=3, n=6, radius=1.
    Uses a full-space candidate pool with NumPy-accelerated greedy set-cover,
    then applies local search (removal + swap) to minimize codeword count.
    """
    if q != 3 or n != 6 or radius != 1:
        return _fallback_construct(q, n, radius)

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # --- Step 1: Enumerate all 729 words and precompute adjacency ---
    # Represent each word as an integer 0..728 (base-3 encoding)
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    all_indices = list(range(729))
    all_words = set(all_indices)

    # Precompute neighbor set for each of the 729 words (Hamming distance <= 1)
    # Each word covers itself + 6*2 = 12 neighbors = 13 total
    neighbors = []
    for idx in all_indices:
        w = idx_to_word(idx)
        covered = {idx}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w[:]
                new_w[pos] = val
                covered.add(word_to_idx(new_w))
        neighbors.append(covered)

    # Build word -> candidates map (which codewords cover this word)
    word_to_cands = {w: set() for w in all_indices}
    for i, cset in enumerate(neighbors):
        for w in cset:
            word_to_cands[w].add(i)

    # --- Step 2: Greedy set cover over full 729-word pool ---
    uncovered = set(all_indices)
    selected = []
    available = set(all_indices)

    while uncovered:
        if time.time() > deadline:
            break

        best_candidate = None
        best_count = -1

        # Count uncovered words covered by each available candidate
        # Use vectorized approach: for each uncovered word, increment count for its cands
        # But 729 candidates * 13 neighbors is small; direct loop is fine
        for i in available:
            cnt = len(neighbors[i] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_candidate = i

        if best_candidate is None:
            break

        selected.append(best_candidate)
        available.discard(best_candidate)
        uncovered -= neighbors[best_candidate]

    # --- Step 3: Local search — remove redundant, then 1-swap moves ---
    def cover_count_map(sel):
        cc = np.zeros(729, dtype=np.int32)
        for i in sel:
            for w in neighbors[i]:
                cc[w] += 1
        return cc

    def is_valid(sel):
        if not sel:
            return False
        cc = cover_count_map(sel)
        return np.all(cc >= 1)

    # Phase A: Remove redundant codewords (iterate to fixpoint)
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        cc = cover_count_map(selected)
        for i in list(selected):
            # i is redundant if no word it covers has count == 1
            if np.all(cc[list(neighbors[i])] > 1):
                selected.remove(i)
                changed = True
                break

    # Phase B: 1-swap local search — try replacing one codeword with another
    # to reduce total count. We try to find a swap that reduces size by 1.
    # Actually, 1-swap keeps size same. To reduce size, we need removal.
    # Instead: try adding a codeword then removing two, net -1.
    # Simpler: just do more removal passes with randomized ordering.

    # Phase C: Randomized restarts of greedy + removal to find better solutions
    best_size = len(selected)
    best_selected = list(selected)

    # Try multiple greedy orders (randomized tie-breaking)
    rng = random.Random(42)
    num_restarts = 0
    while time.time() < deadline - 20:  # leave 20s for final local search
        num_restarts += 1
        # Randomized greedy
        uncovered = set(all_indices)
        sel = []
        avail = set(all_indices)
        
        while uncovered:
            if time.time() > deadline - 20:
                break
            best_c = None
            best_cnt = -1
            # Shuffle for tie-breaking diversity
            for i in avail:
                cnt = len(neighbors[i] & uncovered)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = i
                elif cnt == best_cnt and best_c is not None:
                    # Random tie-break
                    if rng.random() < 0.5:
                        best_c = i
            if best_c is None:
                break
            sel.append(best_c)
            avail.discard(best_c)
            uncovered -= neighbors[best_c]

        # Removal pass
        changed2 = True
        while changed2:
            changed2 = False
            if time.time() > deadline - 20:
                break
            cc = cover_count_map(sel)
            for i in list(sel):
                if np.all(cc[list(neighbors[i])] > 1):
                    sel.remove(i)
                    changed2 = True
                    break

        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = list(sel)

    selected = best_selected

    # Final removal pass on best solution
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        cc = cover_count_map(selected)
        for i in list(selected):
            if np.all(cc[list(neighbors[i])] > 1):
                selected.remove(i)
                changed = True
                break

    # --- Step 4: Validate and return ---
    if not is_valid(selected):
        # Emergency: return all 729 (valid but large)
        selected = all_indices

    result = [idx_to_word(i) for i in selected]
    return result

def _fallback_construct(q, n, radius):
    # Simple random greedy for other cases
    import random
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    uncovered = set(all_words)
    selected = []
    
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
        
    while uncovered:
        w = random.choice(list(uncovered))
        selected.append(w)
        to_remove = []
        for u in uncovered:
            if hamming_dist(w, u) <= radius:
                to_remove.append(u)
        for u in to_remove:
            uncovered.remove(u)
            
    return selected
