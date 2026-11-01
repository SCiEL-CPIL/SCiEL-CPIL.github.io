"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    
    # General fallback
    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    """Optimized construction for K_3(7,1)."""
    
    N = 2187
    all_indices = list(range(N))
    
    # Precompute covers as bitsets (Python ints)
    covers = [0] * N
    for i in range(N):
        digits = [0]*7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    diff = val - digits[pos]
                    new_i = i + diff * (3**pos)
                    mask |= (1 << new_i)
        covers[i] = mask
    
    full_mask = (1 << N) - 1
    
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        selected = []
        
        while uncovered:
            # Find best candidate
            best_count = -1
            best_idx = -1
            best_indices = []
            
            # Optimization: Only check candidates that might be good?
            # For now, check all. 2187 is small enough.
            for i in all_indices:
                cnt = (uncovered & covers[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    best_indices = [i]
                elif cnt == best_count:
                    best_indices.append(i)
            
            if best_count == 0:
                break # Should not happen
                
            if randomize and len(best_indices) > 1:
                idx = random.choice(best_indices)
            else:
                idx = best_idx
            
            selected.append(idx)
            uncovered &= ~covers[idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected, uncovered

    def local_search_remove(code, deadline):
        """
        Try to remove codewords from the code if they are redundant.
        Returns the reduced code.
        """
        if not code:
            return code
        
        # Calculate current coverage
        # coverage_mask = OR of covers[i] for i in code
        coverage_mask = 0
        for i in code:
            coverage_mask |= covers[i]
        
        if coverage_mask != full_mask:
            # Not a valid cover, cannot remove
            return code
        
        # To check if removing 'i' keeps it a valid cover:
        # We need to know if (coverage_mask ^ covers[i]) == full_mask?
        # No, we need to know if the union of the *other* covers is still full_mask.
        # This is hard to update incrementally for arbitrary removals without recounting.
        # However, we can try a heuristic:
        # If a word w is covered ONLY by codeword i, then i is essential.
        # If all words are covered by at least 2 codewords, we can try removing any.
        
        # A faster heuristic for local search:
        # Iterate through codewords. For each, tentatively remove it.
        # Check if the remaining code covers everything.
        # To check coverage efficiently, we can maintain a count of how many codewords cover each word.
        
        # Let's use a count array approach for the local search phase.
        # count[j] = number of codewords in code that cover word j.
        # If min(count) >= 2, we can remove any codeword? No, we need to check specifically.
        # But if we remove i, count[j] decreases by 1 for all j in covers[i].
        # If any count[j] becomes 0, removal is invalid.
        
        # This allows O(1) check for validity of removal if we maintain counts.
        # Update counts: O(15) per removal attempt.
        # Total O(15 * |code|) per pass.
        
        counts = [0] * N
        for i in code:
            mask = covers[i]
            # Iterate through set bits in mask
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Try to remove each codeword in random order
            indices_to_try = list(range(len(code)))
            random.shuffle(indices_to_try)
            
            for idx_in_code in indices_to_try:
                if idx_in_code >= len(code):
                    continue
                c = code[idx_in_code]
                
                # Check if removing c is safe
                # We need to ensure that for all j in covers[c], counts[j] > 1
                # i.e., min(counts[j] for j in covers[c]) > 1
                safe = True
                mask = covers[c]
                m = mask
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if counts[j] <= 1:
                        safe = False
                        break
                    m ^= lsb
                
                if safe:
                    # Remove c
                    # Update counts
                    m = covers[c]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        counts[j] -= 1
                        m ^= lsb
                    
                    code.pop(idx_in_code)
                    changed = True
            
        return code

    def local_search_swap(code, deadline):
        """
        Try to swap a codeword with a non-codeword to enable further reductions or just improve structure.
        This is more complex. We'll stick to removal first.
        """
        return code

    best_code = None
    best_size = N
    
    # 1. Deterministic Greedy
    code, unc = greedy_cover(randomize=False)
    if not unc:
        code = local_search_remove(code, deadline)
        best_code = code
        best_size = len(code)
    
    # 2. Randomized Greedy + Local Search
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        code, unc = greedy_cover(randomize=True, seed=attempts)
        if not unc:
            # Local search to reduce
            code = local_search_remove(code, deadline)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                if best_size <= 186:
                    # We hit the known best. Try to go lower?
                    # Given time constraints, 186 is a very good target.
                    # We can continue searching to see if we find 185, but it's unlikely.
                    # Let's keep searching for a bit more to see if we can beat 186.
                    pass
    
    if best_code is None:
        # Fallback: complete the cover
        code, unc = greedy_cover(randomize=False)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
        best_code = code
        best_size = len(best_code)
    
    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0]*7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)
    
    return result


def _construct_general(q, n, radius, deadline):
    """General fallback for other q, n, radius."""
    import itertools
    
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute covers
    covers = [0] * total
    for i, w in enumerate(all_words):
        mask = 1 << i
        # Generate all words within radius
        # This is expensive for large n, but for small n it's fine.
        # For general case, we might not have time.
        # Let's just do a simple greedy without precomputation for now.
        pass
    
    # Simple greedy
    uncovered = set(range(total))
    selected = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            # Count how many uncovered words are within radius of i
            cnt = 0
            for j in uncovered:
                # Hamming distance
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        # Remove covered words
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)
            
    # Ensure validity
    if uncovered:
        # Add all uncovered
        for j in uncovered:
            selected.append(j)
            
    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
