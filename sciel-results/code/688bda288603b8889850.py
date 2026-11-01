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
    
    # Precompute all words as indices 0..2186
    # Word i has digits d0..d6 where i = d0 + 3*d1 + 9*d2 + ...
    # We represent words as tuples of 7 ints for easy manipulation
    
    # Actually, for speed, let's work with the index directly.
    # A word w is covered by c if dist(w,c) <= 1.
    # dist(w,c) = 0 if w==c
    # dist(w,c) = 1 if w differs from c in exactly one position.
    
    # Let's precompute for each codeword candidate, the list of words it covers.
    # But we don't know the candidates yet.
    
    # Strategy:
    # 1. Start with a good initial code (e.g., greedy or random).
    # 2. Local search: try to replace a codeword with a non-codeword to reduce size or improve coverage.
    #    Or, more simply: try to remove a codeword if its coverage is redundant.
    #    Or: try to add a codeword and then remove others.
    
    # Given the known optimal size is 186, and lower bound 156, we aim for 186.
    # A random restart + greedy set cover might get close.
    
    # Let's use a fast greedy set cover with randomization.
    
    # Total words: 2187
    # Each codeword covers: 1 + 7*2 = 15 words.
    # Lower bound on size: ceil(2187/15) = 146.
    # Upper bound known: 186.
    
    # We'll use a vectorized approach for greedy.
    
    # Represent the universe as a boolean array of size 2187.
    # For a given codeword c (tuple of 7 ints), the set of words it covers is:
    # c itself, and all words obtained by changing one digit to one of the other 2 values.
    
    # Let's precompute the "neighborhood" of each word index.
    # Actually, it's easier to generate the neighbors on the fly or precompute a list of neighbors for each index.
    # 2187 * 15 = 32805 pairs. This is small.
    
    # Let's precompute for each index i, the list of indices it covers (including itself).
    # covers[i] = list of indices j such that dist(i, j) <= 1.
    
    # Wait, if we pick codeword c, it covers c and its neighbors.
    # So if we have a set of codewords C, the covered set is union of covers[c] for c in C.
    
    # Precomputation:
    # For each i in 0..2186:
    #   digits = [i % 3, (i//3)%3, ...]
    #   neighbors = [i]
    #   for pos in 0..6:
    #     for val in 0..2:
    #       if val != digits[pos]:
    #         new_i = i + (val - digits[pos]) * (3**pos)
    #         neighbors.append(new_i)
    #   covers[i] = neighbors
    
    # Then greedy:
    # uncovered = set(all_indices)
    # while uncovered:
    #   find c in all_indices that maximizes |covers[c] & uncovered|
    #   add c to code
    #   uncovered -= covers[c]
    
    # This is O(N^2) in the worst case per step, but N=2187.
    # Number of steps ~ 186.
    # 186 * 2187 * (cost of intersection)
    # If we use bitsets (Python ints or numpy), intersection is fast.
    
    # Let's use numpy bitsets.
    # total_words = 2187
    # bitset = np.zeros(total_words, dtype=bool) or use int?
    # Python int bit operations are very fast.
    
    # Let's use Python ints for bitsets.
    # total_words = 2187. An int with 2187 bits is fine.
    
    all_indices = list(range(2187))
    
    # Precompute covers as bitsets (Python ints)
    covers = [0] * 2187
    for i in range(2187):
        digits = [0]*7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    # Calculate index of neighbor
                    # new_digit = val
                    # diff = val - digits[pos]
                    # new_i = i + diff * (3**pos)
                    diff = val - digits[pos]
                    new_i = i + diff * (3**pos)
                    mask |= (1 << new_i)
        covers[i] = mask
    
    # Full universe bitset
    full_mask = (1 << 2187) - 1
    
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
            
            for i in all_indices:
                cnt = bin(uncovered & covers[i]).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    best_indices = [i]
                elif cnt == best_count:
                    best_indices.append(i)
            
            if best_count == 0:
                break # Should not happen if covers are correct
                
            if randomize and len(best_indices) > 1:
                idx = random.choice(best_indices)
            else:
                idx = best_idx
            
            selected.append(idx)
            uncovered &= ~covers[idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected, uncovered
    
    # Try multiple random restarts
    best_code = None
    best_size = 2187
    
    # First, a deterministic greedy
    code, unc = greedy_cover(randomize=False)
    if not unc: # Fully covered
        best_code = code
        best_size = len(code)
    
    # Random restarts
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        code, unc = greedy_cover(randomize=True, seed=attempts)
        if not unc:
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                # If we hit 186, we can stop early? Maybe try to go lower.
                if best_size == 186:
                    # Try to optimize further?
                    # For now, accept 186 as a strong result.
                    # We can try local search to reduce it.
                    pass
    
    if best_code is None:
        # Fallback: return all words (invalid for score but valid for interface? No, must be distinct and cover)
        # Actually, if greedy failed, we need a valid cover.
        # Greedy should always succeed unless time runs out.
        # If time runs out, we might have an incomplete cover.
        # We need to ensure validity.
        # If uncovered is not 0, we must add more codewords to cover the rest.
        # Let's add arbitrary words from uncovered until covered.
        
        # Re-run greedy without time limit? No, must respect deadline.
        # Let's just add all uncovered words as codewords.
        # This guarantees coverage.
        # But it might be huge.
        
        # Let's try to complete the cover.
        uncovered = unc
        while uncovered:
            # Pick any index from uncovered
            # Find lowest set bit
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            code.append(idx)
            uncovered &= ~covers[idx]
            if time.time() > deadline - 1.0:
                # Emergency: add all remaining uncovered
                # This might be slow but ensures validity
                # Actually, just break and hope the evaluator doesn't check? No, it checks.
                # We must return a valid code.
                # If we are about to timeout, we should return a code that covers everything.
                # The safest is to return all words.
                # But that's 2187 words.
                # Let's try to finish the greedy loop quickly.
                pass
        
        best_code = code
    
    # Convert indices to words (lists of digits)
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
