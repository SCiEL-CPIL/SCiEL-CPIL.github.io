<<<<<<< SEARCH
"""Fresh design: Simulated Annealing with swap moves for ternary covering codes."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import math
import itertools


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 108.0  # safety margin under 120s

    total = q ** n
    pos_step = [q ** (n - 1 - i) for i in range(n)]

    def word_to_idx(word):
        idx = 0
        for i, val in enumerate(word):
            idx += val * pos_step[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = (idx // pos_step[i]) % q
        return word

    def build_cover(codewords):
        """Return coverage count array."""
        cover = np.zeros(total, dtype=np.int32)
        for w in codewords:
            idx = word_to_idx(w)
            cover[idx] += 1
            if radius >= 1:
                for i in range(n):
                    cur_sym = w[i]
                    step = pos_step[i]
                    base = idx - cur_sym * step
                    for v in range(q):
                        if v != cur_sym:
                            cover[base + v * step] += 1
        return cover

    def is_valid_cover(cover):
        return cover.min() >= 1

    # Start with known good construction
    # For q=3, n=8, radius=1, the tetracode product is a strong start.
    # Tetracode is a [4,2,3] code over GF(3).
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Repeat tetracode structure? No, the product construction is:
        # If C1 is a code over GF(q) length n1 and C2 is a code over GF(q) length n2,
        # then C1 x C2 is a code over GF(q) length n1+n2.
        # However, the "tetracode" is often used in a different way for covering codes.
        # Let's stick to the seed from the previous program which was:
        # seed = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        # This is C_tetracode x {0..q-1}^(n-4).
        # This is valid because d(C x D) = min(d(C), d(D)) is not quite right for covering radius,
        # but K_q(n1+n2, r) <= K_q(n1, r1) * K_q(n2, r2) if r1+r2=r? No.
        # Actually, if C covers {0..q-1}^n1 with radius r1 and D covers {0..q-1}^n2 with radius r2,
        # then C x D covers {0..q-1}^(n1+n2) with radius r1+r2.
        # Here we want radius 1. If we use Tetracode (radius 1? No, Tetracode is perfect? No.)
        # Tetracode is a [4,2,3] code. Its covering radius is 1?
        # Distance 3 means min distance 3. Covering radius is 1.
        # So Tetracode covers GF(3)^4 with radius 1.
        # The set {0..q-1}^(n-4) covers GF(3)^(n-4) with radius 0.
        # So the product covers GF(3)^n with radius 1+0=1.
        # Size is 9 * 3^(n-4). For n=8, size is 9 * 81 = 729.
        # Wait, the previous program used this and got 729?
        # The task says best known is 486.
        # The previous program's score was 1.0, which is the seed.
        # The seed program likely returned this 729 size code?
        # Let's check the previous code's output size.
        # The previous code started with this seed and then did local search.
        # If the previous score was 1.0, and the seed is 729, then the best known in the population might be 729?
        # Or maybe the seed was different?
        # Let's assume the previous seed was the standard product construction.
        
        seed = []
        # Generate tetracode words
        t_words = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Generate tails
        tails = list(itertools.product(range(q), repeat=n - 4))
        for h in t_words:
            for tail in tails:
                seed.append(h + list(tail))
    else:
        # Fallback: Random or simple grid
        seed = [[random.randrange(q) for _ in range(n)] for _ in range(100)]
        # Ensure validity? Random might not be valid.
        # For general q,n,r, this is hard.
        # But the task is specific: q=3, n=8, r=1.
        # We assume the evaluator passes these parameters.
        # If not, we return a large random set? No, must be valid.
        # Let's just use the product construction if possible, else brute force small?
        # Given the constraints, we assume q=3, n=8.
        pass

    best = list(seed)
    cover = build_cover(best)
    
    if not is_valid_cover(cover):
        # Fallback: Add random words until valid (expensive but safe)
        while not is_valid_cover(cover):
            w = [random.randrange(q) for _ in range(n)]
            if w not in best:
                best.append(w)
                idx = word_to_idx(w)
                cover[idx] += 1
                if radius >= 1:
                    for i in range(n):
                        cur_sym = w[i]
                        step = pos_step[i]
                        base = idx - cur_sym * step
                        for v in range(q):
                            if v != cur_sym:
                                cover[base + v * step] += 1

    best_len = len(best)
    
    # Helper to update cover when adding/removing a word
    def update_cover_add(cover, w):
        idx = word_to_idx(w)
        cover[idx] += 1
        if radius >= 1:
            for i in range(n):
                cur_sym = w[i]
                step = pos_step[i]
                base = idx - cur_sym * step
                for v in range(q):
                    if v != cur_sym:
                        cover[base + v * step] += 1

    def update_cover_remove(cover, w):
        idx = word_to_idx(w)
        cover[idx] -= 1
        if radius >= 1:
            for i in range(n):
                cur_sym = w[i]
                step = pos_step[i]
                base = idx - cur_sym * step
                for v in range(q):
                    if v != cur_sym:
                        cover[base + v * step] -= 1

    def try_delete_all(cover, codewords):
        """Aggressively delete any redundant codeword."""
        changed = True
        while changed:
            if time.time() > DEADLINE - 2:
                break
            changed = False
            for ci in range(len(codewords)):
                w = codewords[ci]
                idx = word_to_idx(w)
                if cover[idx] > 1:
                    ok = True
                    for i in range(n):
                        cur_sym = w[i]
                        step = pos_step[i]
                        base = idx - cur_sym * step
                        for v in range(q):
                            if v != cur_sym:
                                if cover[base + v * step] < 2:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        update_cover_remove(cover, w)
                        codewords.pop(ci)
                        changed = True
                        break

    # Simulated Annealing
    T_start = 1.0
    T_end = 0.001
    alpha = 0.99
    T = T_start
    
    best_code = list(best)
    best_cover = cover.copy()
    
    current_code = list(best)
    current_cover = cover.copy()
    current_len = len(current_code)
    
    # Precompute list of all words? No, too large (6561).
    # But we can generate random words.
    
    while time.time() < DEADLINE - 2:
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.4 and len(current_code) > 1:
            # Move: Delete a random codeword
            ci = random.randrange(len(current_code))
            w = current_code[ci]
            update_cover_remove(current_cover, w)
            if is_valid_cover(current_cover):
                # Accept
                current_code.pop(ci)
                current_len -= 1
                if current_len < best_len:
                    best_len = current_len
                    best_code = list(current_code)
                    best_cover = current_cover.copy()
            else:
                # Reject
                update_cover_add(current_cover, w)
        
        elif move_type < 0.7:
            # Move: Insert a random codeword
            # Check if it's already there? 
            # To avoid duplicates, we can track a set of indices.
            # For simplicity, we allow duplicates in the list? No, "codewords must be distinct".
            # So we must check.
            # Using a set of indices for O(1) lookup.
            # We don't maintain a set in the main loop for speed, so we do a linear check?
            # Or we can just try and if it's a duplicate, it's a no-op effectively?
            # No, if we add a duplicate, cover counts go up, but it's not a valid code if we return duplicates.
            # The evaluator checks "codewords must be distinct".
            # So we must ensure distinctness.
            # Let's maintain a set of indices for the current code.
            # This adds overhead.
            # Alternative: Just generate random words and check against a set.
            
            # Let's refactor to use a set of indices for current_code.
            # This might be too slow if we recreate it.
            # Let's just do a linear check for now, n=8, size ~500. 500 checks is fast.
            
            new_word = [random.randrange(q) for _ in range(n)]
            # Check distinctness
            is_dup = False
            for w in current_code:
                if w == new_word:
                    is_dup = True
                    break
            
            if not is_dup:
                update_cover_add(current_cover, new_word)
                current_code.append(new_word)
                new_len = len(current_code)
                
                # Acceptance
                # Delta = new_len - current_len = 1
                # If T is high, we accept.
                # We want to escape local minima, so we accept worse solutions.
                if random.random() < math.exp(-1.0 / T):
                    current_len = new_len
                    # Try to delete immediately to see if we can get back to same or better size
                    try_delete_all(current_cover, current_code)
                    if len(current_code) < best_len:
                        best_len = len(current_code)
                        best_code = list(current_code)
                        best_cover = current_cover.copy()
                else:
                    # Reject
                    current_cover[ word_to_idx(new_word) ] -= 1 # Manual revert? 
                    # Better to use update_cover_remove
                    update_cover_remove(current_cover, new_word)
                    current_code.pop()
                    current_len = len(current_code)
            # If is_dup, do nothing (move failed)
            
        else:
            # Move: Swap a random codeword with a random new word
            if len(current_code) > 0:
                ci = random.randrange(len(current_code))
                old_word = current_code[ci]
                new_word = [random.randrange(q) for _ in range(n)]
                
                # Check if new_word is same as old_word or already in code (excluding old_word)
                # If new_word == old_word, no change.
                # If new_word is in code elsewhere, we'd have duplicates.
                
                # Check distinctness
                is_dup = False
                for j, w in enumerate(current_code):
                    if j != ci and w == new_word:
                        is_dup = True
                        break
                
                if not is_dup and new_word != old_word:
                    # Remove old, add new
                    update_cover_remove(current_cover, old_word)
                    update_cover_add(current_cover, new_word)
                    current_code[ci] = new_word
                    
                    # Size is unchanged.
                    # Accept if valid? It is valid if cover >= 1.
                    # Since we removed old and added new, and old was valid,
                    # new might leave holes if new doesn't cover what old covered and no other code covers those holes.
                    if is_valid_cover(current_cover):
                        # Accept
                        # Try to delete
                        try_delete_all(current_cover, current_code)
                        if len(current_code) < best_len:
                            best_len = len(current_code)
                            best_code = list(current_code)
                            best_cover = current_cover.copy()
                    else:
                        # Reject
                        update_cover_remove(current_cover, new_word)
                        update_cover_add(current_cover, old_word)
                        current_code[ci] = old_word
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end

    # Final cleanup
    try_delete_all(best_cover, best_code)
    
    return best_code

# EVOLVE-BLOCK-END
=======
"""Greedy set cover + iterative local search for ternary covering codes."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import math
import itertools


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 108.0  # safety margin under 120s

    total = q ** n
    pos_step = [q ** (n - 1 - i) for i in range(n)]

    def word_to_idx(word):
        idx = 0
        for i, val in enumerate(word):
            idx += val * pos_step[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = (idx // pos_step[i]) % q
        return word

    def get_neighbors(idx):
        """Yield all indices within Hamming distance <= radius from idx."""
        yield idx
        if radius >= 1:
            for i in range(n):
                cur_sym = (idx // pos_step[i]) % q
                step = pos_step[i]
                base = idx - cur_sym * step
                for v in range(q):
                    if v != cur_sym:
                        yield base + v * step

    def build_cover_from_indices(indices):
        """Return coverage count array from a list of indices."""
        cover = np.zeros(total, dtype=np.int32)
        for idx in indices:
            for nb in get_neighbors(idx):
                cover[nb] += 1
        return cover

    def is_valid_cover(cover):
        return cover.min() >= 1

    def greedy_set_cover():
        """Greedy set cover: repeatedly pick the word that covers the most uncovered points."""
        uncovered = np.ones(total, dtype=np.int32)
        selected = []
        
        while uncovered.sum() > 0:
            if time.time() > DEADLINE - 30:
                break
            
            best_idx = -1
            best_gain = 0
            # Sample a subset of candidate words for speed
            sample_size = min(200, total)
            candidates = random.sample(range(total), sample_size)
            
            for idx in candidates:
                gain = 0
                for nb in get_neighbors(idx):
                    gain += uncovered[nb]
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx == -1 or best_gain == 0:
                # Fallback: pick a random uncovered point's neighborhood
                uncov_list = np.where(uncovered > 0)[0]
                if len(uncov_list) == 0:
                    break
                target = random.choice(uncov_list.tolist())
                best_idx = target
                best_gain = 1
            
            selected.append(best_idx)
            for nb in get_neighbors(best_idx):
                uncovered[nb] = 0
        
        return selected

    def local_search_remove(codeword_indices, cover):
        """Iteratively remove redundant codewords."""
        changed = True
        while changed:
            if time.time() > DEADLINE - 5:
                break
            changed = False
            for ci in range(len(codeword_indices)):
                idx = codeword_indices[ci]
                # Check if this codeword is redundant
                redundant = True
                for nb in get_neighbors(idx):
                    if cover[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    for nb in get_neighbors(idx):
                        cover[nb] -= 1
                    codeword_indices.pop(ci)
                    changed = True
                    break
        return codeword_indices

    def randomized_local_search(codeword_indices, cover, iterations=5000):
        """Randomized local search with add/remove/swap moves."""
        current_indices = list(codeword_indices)
        current_cover = cover.copy()
        current_set = set(current_indices)
        best_indices = list(current_indices)
        best_len = len(current_indices)
        
        for _ in range(iterations):
            if time.time() > DEADLINE - 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.3 and len(current_indices) > 1:
                # Remove a random codeword
                ci = random.randrange(len(current_indices))
                idx = current_indices[ci]
                for nb in get_neighbors(idx):
                    current_cover[nb] -= 1
                
                if is_valid_cover(current_cover):
                    current_indices.pop(ci)
                    current_set.discard(idx)
                    if len(current_indices) < best_len:
                        best_len = len(current_indices)
                        best_indices = list(current_indices)
                else:
                    for nb in get_neighbors(idx):
                        current_cover[nb] += 1
            
            elif move_type < 0.6:
                # Add a random codeword
                new_idx = random.randrange(total)
                if new_idx not in current_set:
                    for nb in get_neighbors(new_idx):
                        current_cover[nb] += 1
                    current_indices.append(new_idx)
                    current_set.add(new_idx)
                    
                    # Try to remove redundant codewords
                    local_search_remove(current_indices, current_cover)
                    current_set = set(current_indices)
                    
                    if len(current_indices) < best_len:
                        best_len = len(current_indices)
                        best_indices = list(current_indices)
                    else:
                        # Revert
                        for nb in get_neighbors(new_idx):
                            current_cover[nb] -= 1
                        current_indices.pop()
                        current_set.discard(new_idx)
            
            else:
                # Swap: remove one, add another
                if len(current_indices) > 0:
                    ci = random.randrange(len(current_indices))
                    old_idx = current_indices[ci]
                    new_idx = random.randrange(total)
                    
                    if new_idx != old_idx and new_idx not in current_set:
                        # Remove old
                        for nb in get_neighbors(old_idx):
                            current_cover[nb] -= 1
                        # Add new
                        for nb in get_neighbors(new_idx):
                            current_cover[nb] += 1
                        
                        if is_valid_cover(current_cover):
                            current_indices[ci] = new_idx
                            current_set.discard(old_idx)
                            current_set.add(new_idx)
                            
                            # Try to remove redundant
                            local_search_remove(current_indices, current_cover)
                            current_set = set(current_indices)
                            
                            if len(current_indices) < best_len:
                                best_len = len(current_indices)
                                best_indices = list(current_indices)
                        else:
                            # Revert
                            for nb in get_neighbors(new_idx):
                                current_cover[nb] -= 1
                            for nb in get_neighbors(old_idx):
                                current_cover[nb] += 1
                            current_indices[ci] = old_idx
        
        return best_indices, best_len

    # Phase 1: Greedy set cover
    initial_indices = greedy_set_cover()
    initial_cover = build_cover_from_indices(initial_indices)
    
    if not is_valid_cover(initial_cover):
        # Fallback: add words until valid
        while not is_valid_cover(initial_cover):
            idx = random.randrange(total)
            if idx not in initial_indices:
                initial_indices.append(idx)
                for nb in get_neighbors(idx):
                    initial_cover[nb] += 1
    
    # Phase 2: Aggressive removal
    initial_indices = local_search_remove(initial_indices, initial_cover)
    initial_cover = build_cover_from_indices(initial_indices)
    
    best_indices = list(initial_indices)
    best_len = len(best_indices)
    
    # Phase 3: Iterative randomized local search with restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Perturb: remove some random codewords and re-add via greedy
        perturbed = list(best_indices)
        if len(perturbed) > 10:
            remove_count = max(1, len(perturbed) // 10)
            remove_indices = random.sample(range(len(perturbed)), remove_count)
            for ri in sorted(remove_indices, reverse=True):
                perturbed.pop(ri)
        
        # Re-cover if needed
        perturbed_cover = build_cover_from_indices(perturbed)
        if not is_valid_cover(perturbed_cover):
            # Add words greedily until valid
            while not is_valid_cover(perturbed_cover) and time.time() < DEADLINE - 10:
                uncov_list = np.where(perturbed_cover == 0)[0]
                if len(uncov_list) == 0:
                    break
                target = random.choice(uncov_list.tolist())
                # Find a word that covers target
                best_candidate = target
                for _ in range(10):
                    candidate = random.randrange(total)
                    # Check if it covers target
                    dist = 0
                    c_word = idx_to_word(candidate)
                    t_word = idx_to_word(target)
                    for i in range(n):
                        if c_word[i] != t_word[i]:
                            dist += 1
                            if dist > radius:
                                break
                    if dist <= radius:
                        best_candidate = candidate
                        break
                
                if best_candidate not in perturbed:
                    perturbed.append(best_candidate)
                    for nb in get_neighbors(best_candidate):
                        perturbed_cover[nb] += 1
        
        # Local search
        new_indices, new_len = randomized_local_search(perturbed, perturbed_cover, iterations=3000)
        
        if new_len < best_len:
            best_len = new_len
            best_indices = new_indices
    
    # Final cleanup
    final_cover = build_cover_from_indices(best_indices)
    best_indices = local_search_remove(best_indices, final_cover)
    
    # Convert indices back to words
    result = [idx_to_word(idx) for idx in best_indices]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for w in result:
        w_tuple = tuple(w)
        if w_tuple not in seen:
            seen.add(w_tuple)
            unique_result.append(w)
    
    return unique_result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
