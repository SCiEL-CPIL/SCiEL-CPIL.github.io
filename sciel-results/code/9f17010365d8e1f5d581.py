import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    start_time = time.time()
    time_limit = 145.0

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

    # Precompute neighbors for all 6561 words
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig_val

    all_words = [idx_to_word(i) for i in range(total_points)]
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total_points)]

    def is_valid_cover(chosen_indices):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return np.all(cover_count > 0)

    # Tetracode words as tuples
    tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
    tetracode_set = set(tetracode)

    # Automorphism group of tetracode: coordinate permutations + symbol permutations that preserve the code
    # Tetracode is the [4,2,2] code. Its automorphism group acts on the 4 coordinates.
    # We'll use a simpler approach: define cosets by the first 4 coordinates being in tetracode
    # and the last 4 coordinates being in tetracode.
    # A "coset" in this context: fix the first 4 coords to a specific tetracode word,
    # and vary the last 4 coords over all tetracode words (or vice versa).
    
    # Actually, let's use a different algebraic structure:
    # The tetracode has 9 words. The product code has 81 words.
    # We want to grow to ~486 words. 
    # Strategy: Start with the 81-word product code.
    # Then, for each of the 9 tetracode words in the first half, we have 9 words in the second half.
    # We can "expand" by allowing the second half to be any word, but that's 3^4=81 per first-half word -> 9*81=729 words.
    # That's too many. We want 486.
    
    # Let's try a different algebraic approach:
    # Use the fact that the covering radius is 1.
    # A word is covered if it's within distance 1 of a codeword.
    # The "shadow" of a code is the set of words at distance 2.
    
    # Let's stick to the coset idea but make it more concrete:
    # Define "blocks" based on the first 4 coordinates.
    # There are 3^4 = 81 possible first-4-coordinate patterns.
    # For each pattern p, let C_p be the set of codewords whose first 4 coords are p.
    # If p is in tetracode, we start with |C_p| = 9 (the tetracode words for the last 4 coords).
    # If p is not in tetracode, we start with |C_p| = 0.
    # Total start: 9 * 9 = 81.
    # We want to add words. We can add words to any C_p.
    # A local move: pick a block p, remove one word from C_p, add one word to C_p' (possibly same or different).
    # To keep size fixed, we do swaps: remove from C_p, add to C_p'.
    
    # But the directive says "restrict moves to cosets of the code's automorphism group".
    # Let's interpret "coset" as: the set of words that differ from a given tetracode product word
    # in the last 4 coordinates only, or in the first 4 coordinates only.
    # Actually, a simpler interpretation: 
    # The tetracode is linear. Its automorphism group includes coordinate permutations.
    # Let's define orbits under the automorphism group of the product code.
    # This is getting complicated. Let's simplify:
    
    # New Strategy:
    # 1. Start with the 81-word tetracode product.
    # 2. Compute the "defect" of each word: how many words are NOT covered if we remove this codeword?
    #    Actually, compute for each word w, the number of codewords within distance 1 of w.
    # 3. Use a local search that tries to swap codewords to reduce the total size while maintaining coverage.
    #    But we want to REDUCE size, not fix it.
    #    Wait, the current program starts with 81 and ADDS words to cover everything.
    #    Then it tries to REMOVE words.
    #    The problem is that greedy addition might add too many, and removal is hard.
    
    # Let's try a different angle: 
    # Instead of adding then removing, let's use a "surgical" approach.
    # 1. Start with the 81-word code.
    # 2. Identify all uncovered words.
    # 3. For each uncovered word u, find the best codeword c to add that covers u and maximizes coverage of other uncovered words.
    # 4. But also, for each existing codeword c, check if it can be removed.
    # 5. Iterate: add one, remove one, etc.
    
    # This is essentially what the current program does. The key is to make the local search smarter.
    
    # Let's implement the coset-restricted search as follows:
    # Partition the 6561 words into "cosets" based on their first 4 coordinates.
    # There are 81 cosets, each of size 81 (the last 4 coordinates vary over all 3^4 words).
    # Within each coset, the words are "similar" in the first half.
    # A local move: pick a coset, and swap a word in it with a word in another coset.
    # But to make it more structured, let's restrict swaps to within the same "type" of coset.
    
    # Actually, let's use a simpler but effective method:
    # "Iterated Local Search" with a specific neighborhood.
    # Neighborhood: 
    #   - Remove a codeword and add a different one (swap).
    #   - The new codeword must be "close" to the removed one (Hamming distance <= 2).
    #   - This restricts the search space significantly.
    
    # Let's implement this.
    
    def compute_cover_count(chosen_indices):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return cover_count

    def local_search_swap(chosen_indices, cover_count):
        """
        Try to swap codewords to maintain coverage while potentially allowing future removals.
        We'll try to increase the minimum coverage count, which makes removal easier.
        """
        current = list(chosen_indices)
        cc = cover_count.copy()
        improved = True
        max_rounds = 100
        round_num = 0

        while improved and round_num < max_rounds and time.time() - start_time < time_limit - 10.0:
            improved = False
            round_num += 1
            min_cov = np.min(cc)
            
            # Try each codeword
            for i in range(len(current)):
                if time.time() - start_time > time_limit - 10.0:
                    break
                ci = current[i]
                # Try to find a replacement cj such that:
                # 1. cj is not in current
                # 2. After swapping ci -> cj, all points remain covered
                # 3. The minimum coverage count increases (or stays same, but we prefer increase)
                
                # Candidates: words within distance 2 of ci
                # Distance 0: ci itself (skip)
                # Distance 1: neighbors[ci] (16 words)
                # Distance 2: words at distance 2 from ci
                # Let's precompute distance-2 neighbors? That's 16*16 = 256 words per ci. Too many.
                # Let's just use distance 1 first.
                
                candidates = [neighbors[ci][j] for j in range(1, neighbors_per_word)]
                random.shuffle(candidates)
                
                best_cj = -1
                best_min_cov = min_cov
                
                for cj in candidates:
                    if cj in current:
                        continue
                    # Check if swap is valid
                    cc[neighbors[ci]] -= 1
                    cc[neighbors[cj]] += 1
                    if np.all(cc > 0):
                        new_min_cov = np.min(cc)
                        if new_min_cov > best_min_cov:
                            best_min_cov = new_min_cov
                            best_cj = cj
                    cc[neighbors[ci]] += 1
                    cc[neighbors[cj]] -= 1
                
                if best_cj >= 0:
                    cc[neighbors[ci]] -= 1
                    cc[neighbors[best_cj]] += 1
                    current[i] = best_cj
                    improved = True
                    break  # Restart the loop after a swap
        
        return current, cc

    def try_remove(chosen_indices, cover_count):
        """
        Try to remove codewords one by one.
        """
        current = list(chosen_indices)
        cc = cover_count.copy()
        removed = True
        while removed and time.time() - start_time < time_limit - 10.0:
            removed = False
            for i in range(len(current)):
                ci = current[i]
                exposed = neighbors[ci]
                would_uncover = exposed[cc[exposed] == 1]
                if len(would_uncover) == 0:
                    cc[neighbors[ci]] -= 1
                    current.pop(i)
                    removed = True
                    break
        return current, cc

    # Main algorithm
    # 1. Start with tetracode product (81 words)
    initial_code = []
    for t1 in tetracode:
        for t2 in tetracode:
            word = list(t1) + list(t2)
            idx = word_to_idx(word)
            initial_code.append(idx)
    
    # 2. Greedy completion: add words to cover all points
    cc = compute_cover_count(initial_code)
    while not np.all(cc > 0) and time.time() - start_time < time_limit - 20.0:
        uncovered = np.where(cc == 0)[0]
        if len(uncovered) == 0:
            break
        # Find the word that covers the most uncovered points
        best_idx = -1
        best_count = -1
        for idx in range(total_points):
            if idx in initial_code:
                continue
            cnt = np.sum(cc[neighbors[idx]] == 0)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx < 0:
            break
        initial_code.append(best_idx)
        cc[neighbors[best_idx]] += 1
    
    # 3. Iterated local search:
    #    - Try to remove words
    #    - If stuck, do a swap to increase min coverage
    #    - Repeat
    best_code = list(initial_code)
    best_cc = cc.copy()
    
    for iteration in range(50):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        # Try to remove
        code, cc = try_remove(best_code, best_cc)
        if len(code) < len(best_code):
            best_code = code
            best_cc = cc
            continue
        
        # Try to improve via swaps
        code, cc = local_search_swap(best_code, best_cc)
        
        # After swaps, try to remove again
        code, cc = try_remove(code, cc)
        if len(code) < len(best_code):
            best_code = code
            best_cc = cc
    
    # Final validity check
    if not is_valid_cover(best_code):
        # Fallback: use the initial code
        best_code = initial_code
    
    return [all_words[i] for i in best_code]
