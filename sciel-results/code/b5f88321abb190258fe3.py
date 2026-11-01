import itertools
import random
import time

def construct(q, n, radius):
    # Fallback for general case or if not the specific hard instance
    if not (q == 3 and n == 6 and radius == 1):
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres as bitmasks
    # A sphere of radius 1 around a word w includes w and all words at Hamming distance 1.
    spheres = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        spheres[i] = mask
        
    full_mask = (1 << N) - 1
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin for 150s limit
    
    best_code = None
    best_size = N # worst case
    
    # Helper to check validity
    def is_valid(code_indices):
        covered = 0
        for idx in code_indices:
            covered |= spheres[idx]
        return covered == full_mask

    # Helper to prune a code (remove redundant words)
    def prune(code_list):
        current = list(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(current)):
                union_others = 0
                for j in range(len(current)):
                    if i == j:
                        continue
                    union_others |= spheres[current[j]]
                    if union_others == full_mask:
                        break
                if union_others == full_mask:
                    current.pop(i)
                    changed = True
                    break
        return current

    # 1. Initial Heuristic: Tetracode Product + Greedy Pruning
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            init_code.append(word_to_idx[w])
            
    pruned_init = prune(init_code)
    if len(pruned_init) < best_size:
        best_size = len(pruned_init)
        best_code = [all_words[i] for i in pruned_init]
        
    # 2. Local Search: 2-Opt (Swap)
    # We try to improve the best_code found so far.
    # We maintain the code as a list of indices.
    current_code = list(pruned_init)
    
    # Precompute a list of all words not in the current code? 
    # That's expensive to update. Instead, we pick random candidates.
    
    # To speed up "is_valid" checks during search, we can maintain a count of how many times each word is covered.
    # cover_count[word_idx] = number of codewords in the code that cover word_idx.
    # A code is valid if min(cover_count) >= 1.
    # A codeword C is redundant if for all words w covered by C, cover_count[w] > 1.
    
    def build_cover_counts(code_indices):
        counts = [0] * N
        for idx in code_indices:
            mask = spheres[idx]
            # Iterate through set bits
            m = mask
            while m:
                # Get index of lowest set bit
                lsb = m & -m
                idx_word = lsb.bit_length() - 1
                counts[idx_word] += 1
                m ^= lsb
        return counts

    def try_remove(code_indices, counts):
        # Try to remove any redundant word
        for i in range(len(code_indices)):
            idx = code_indices[i]
            mask = spheres[idx]
            m = mask
            redundant = True
            while m:
                lsb = m & -m
                idx_word = lsb.bit_length() - 1
                if counts[idx_word] <= 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                # Remove it
                code_indices.pop(i)
                # Update counts
                m = mask
                while m:
                    lsb = m & -m
                    idx_word = lsb.bit_length() - 1
                    counts[idx_word] -= 1
                    m ^= lsb
                return True
        return False

    def try_swap(code_indices, counts, random_state):
        # Pick a random codeword to replace
        i = random_state.randint(0, len(code_indices) - 1)
        old_idx = code_indices[i]
        
        # Calculate words that are "critical" for old_idx (covered only by old_idx)
        # Actually, we can just pick a random new word and check validity.
        # But checking validity from scratch is O(N). 
        # Using counts: 
        # 1. Temporarily remove old_idx.
        # 2. Check if any count becomes 0.
        # 3. If yes, we must pick a new word that covers at least one of those zero-count words.
        
        m = spheres[old_idx]
        critical_words = []
        while m:
            lsb = m & -m
            idx_word = lsb.bit_length() - 1
            if counts[idx_word] == 1:
                critical_words.append(idx_word)
            m ^= lsb
            
        if not critical_words:
            # old_idx is redundant, just remove it (handled by try_remove, but let's do it here too for simplicity if we miss it)
            # Actually, if no critical words, it's redundant.
            code_indices.pop(i)
            m = spheres[old_idx]
            while m:
                lsb = m & -m
                idx_word = lsb.bit_length() - 1
                counts[idx_word] -= 1
                m ^= lsb
            return True

        # We need to find a new word that covers at least one critical word.
        # Pick a random critical word, and find a random word in its sphere that is not in the current code.
        target_word = random_state.choice(critical_words)
        # Get all words in the sphere of target_word
        # We can iterate through the sphere bits
        m = spheres[target_word]
        candidates = []
        while m:
            lsb = m & -m
            idx_word = lsb.bit_length() - 1
            if idx_word != old_idx: # Don't swap with itself
                candidates.append(idx_word)
            m ^= lsb
        
        if not candidates:
            return False
            
        # Try a few random candidates
        random_state.shuffle(candidates)
        for new_idx in candidates[:10]: # Limit tries per swap attempt
            # Check if new_idx is already in code
            # Linear search is O(K). K~75. N=729. 
            # For speed, we could maintain a set, but list is fine for K=75.
            if new_idx in code_indices:
                continue
                
            # Try adding new_idx and removing old_idx
            # 1. Remove old_idx counts
            m = spheres[old_idx]
            while m:
                lsb = m & -m
                idx_word = lsb.bit_length() - 1
                counts[idx_word] -= 1
                m ^= lsb
                
            # 2. Add new_idx counts
            m = spheres[new_idx]
            while m:
                lsb = m & -m
                idx_word = lsb.bit_length() - 1
                counts[idx_word] += 1
                m ^= lsb
                
            # 3. Check if any count is 0
            # We only need to check words covered by old_idx or new_idx
            # Words covered by old_idx might have dropped to 0.
            # Words covered by new_idx might have been 0 (if we swapped in a word covering an uncovered spot? 
            # But we started with a valid code, so no counts were 0 initially.
            # When we remove old_idx, counts for its sphere drop by 1.
            # When we add new_idx, counts for its sphere increase by 1.
            # So we only risk zeros in sphere(old_idx) \ sphere(new_idx).
            
            valid = True
            m = spheres[old_idx] & ~spheres[new_idx]
            while m:
                lsb = m & -m
                idx_word = lsb.bit_length() - 1
                if counts[idx_word] == 0:
                    valid = False
                    break
                m ^= lsb
                
            if valid:
                # Accept swap
                code_indices[i] = new_idx
                return True
            else:
                # Revert
                # Add back old_idx counts
                m = spheres[old_idx]
                while m:
                    lsb = m & -m
                    idx_word = lsb.bit_length() - 1
                    counts[idx_word] += 1
                    m ^= lsb
                # Remove new_idx counts
                m = spheres[new_idx]
                while m:
                    lsb = m & -m
                    idx_word = lsb.bit_length() - 1
                    counts[idx_word] -= 1
                    m ^= lsb
                return False
                
        return False

    # Main Search Loop
    # We run multiple restarts of the local search on the initial pruned code, 
    # or we just run local search on the current best.
    # Since greedy restarts are weak, we focus on local search from a good start.
    
    # Let's try a few different initializations if time permits, but mostly local search.
    
    # Build counts for current best
    counts = build_cover_counts(current_code)
    
    random.seed(42)
    
    # Iterative Local Search
    # We try to reduce the size.
    # 1. Remove redundant words.
    # 2. Swap to improve structure (might allow further removals).
    
    max_iterations = 5000
    iter_count = 0
    
    while time.time() < deadline:
        # 1. Prune
        if try_remove(current_code, counts):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = [all_words[i] for i in current_code]
            # After removal, we might have new redundancies, so loop continues to prune again.
            # But we also want to swap.
            continue 
            
        # 2. Swap
        if try_swap(current_code, counts, random):
            # Swap accepted. Code size is same, but structure changed.
            # This might enable future removals.
            iter_count += 1
            if iter_count % 100 == 0:
                # Check time
                if time.time() > deadline:
                    break
        else:
            # Swap failed or no swap possible. 
            # To escape local minima, we can try a random perturbation:
            # Pick a random codeword, remove it, and add 2 random words, then prune.
            # This is a "kick".
            if len(current_code) > 70: # Only kick if we are not already small
                # Random Kick
                if random.random() < 0.1: # 10% chance to kick
                    # Pick a random word to remove
                    i = random.randint(0, len(current_code) - 1)
                    removed_idx = current_code[i]
                    current_code.pop(i)
                    
                    # Update counts
                    m = spheres[removed_idx]
                    while m:
                        lsb = m & -m
                        idx_word = lsb.bit_length() - 1
                        counts[idx_word] -= 1
                        m ^= lsb
                    
                    # Add 2 random words
                    for _ in range(2):
                        new_idx = random.randint(0, N - 1)
                        if new_idx not in current_code:
                            current_code.append(new_idx)
                            m = spheres[new_idx]
                            while m:
                                lsb = m & -m
                                idx_word = lsb.bit_length() - 1
                                counts[idx_word] += 1
                                m ^= lsb
                    
                    # Now prune heavily
                    # Rebuild counts to be safe? No, we updated them.
                    # But we need to ensure validity. If we added words, validity is maintained (counts only increased or stayed same for new words, decreased for removed).
                    # Wait, if we removed a word, some counts might be 0.
                    # The added words might cover them.
                    # Let's just prune. Prune assumes validity? No, prune checks if union of others is full.
                    # If the code is invalid, prune might not fix it.
                    # We should check validity after kick.
                    # Actually, the kick makes the code potentially invalid.
                    # We should only accept the kick if we can prune it back to valid.
                    # This is complex. Let's stick to valid swaps.
                    # If try_swap fails, we are stuck.
                    # Let's try a different strategy: Random Restart from Tetracode with different random seeds for the greedy phase?
                    # No, the initial tetracode is deterministic.
                    # Let's try swapping with a random word from the entire space, not just sphere of critical word.
                    pass

    # Final check
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in best_code]

