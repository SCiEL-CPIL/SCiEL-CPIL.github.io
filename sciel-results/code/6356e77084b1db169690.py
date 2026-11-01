import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
                
    # Invert balls: for each word, which codewords cover it?
    # Use bitsets for fast intersection
    # total codewords is 6561, fits in 103 uint64s or just Python int
    word_to_codewords_bits = [0] * total
    for c in range(total):
        bit = 1 << c
        for w in balls[c]:
            word_to_codewords_bits[w] |= bit

    best_size = total
    best_set = None
    
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 10.0:
        # 1. Greedy
        selected = []
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered):
            counts = uncovered[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.integers(len(candidates))]
            
            selected.append(best_idx)
            uncovered[balls[best_idx]] = False
            
            if time.time() > deadline - 15.0:
                break
        
        if np.any(uncovered):
            continue
            
        selected = np.array(selected, dtype=np.int32)
        
        # 2. Pruning
        cover_count = np.zeros(total, dtype=np.int32)
        cover_count[balls[selected]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            mins = np.min(cover_count[balls[selected]], axis=1)
            removable_mask = mins > 1
            
            if not np.any(removable_mask):
                break
                
            remove_idx = selected[removable_mask]
            selected = selected[~removable_mask]
            
            cover_count[balls[remove_idx]] -= 1
            changed = True
            
        # 3. Local Search
        def try_swaps(selected, cover_count, max_iters=500):
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                if len(selected) == 0:
                    break
                
                # Prune first
                mins = np.min(cover_count[balls[selected]], axis=1)
                removable_mask = mins > 1
                if np.any(removable_mask):
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    continue
                
                # Try 1-for-1 swap
                c_idx = rng.integers(len(selected))
                c = selected[c_idx]
                
                c_words = balls[c]
                unique_mask = cover_count[c_words] == 1
                unique_words = c_words[unique_mask]
                
                if len(unique_words) == 0:
                    continue
                
                # Find candidates using bitsets
                candidates_bits = word_to_codewords_bits[unique_words[0]]
                for w in unique_words[1:]:
                    candidates_bits &= word_to_codewords_bits[w]
                
                # Clear bit for c
                candidates_bits &= ~(1 << c)
                
                if candidates_bits == 0:
                    continue
                
                # Pick a random candidate from the bitset
                # Get list of set bits
                # Efficient way: 
                # while candidates_bits:
                #    lsb = candidates_bits & -candidates_bits
                #    c_prime = lsb.bit_length() - 1
                #    ...
                # For simplicity, if few candidates, convert to list.
                # But bit_length loop is fast.
                
                # Random selection from bitset
                # One way: generate random index, check if set. Repeat.
                # Better: extract bits.
                # Since 6561 bits, we can just loop.
                # But we want random.
                # Let's just pick the first one for speed, or random if we can.
                # Randomness helps escape local minima.
                
                # To pick random:
                # count bits? No.
                # Just iterate and pick with probability?
                # Simpler: get list of indices.
                # It's at most 17 neighbors per word, intersection is small.
                cand_list = []
                temp_bits = candidates_bits
                while temp_bits:
                    lsb = temp_bits & -temp_bits
                    cand_list.append(lsb.bit_length() - 1)
                    temp_bits ^= lsb
                    
                if not cand_list:
                    continue
                
                c_prime = cand_list[rng.integers(len(cand_list))]
                
                # Perform swap
                cover_count[balls[c]] -= 1
                cover_count[balls[c_prime]] += 1
                selected[c_idx] = c_prime
                
                # Prune again
                changed = True
                while changed and time.time() < deadline - 5.0:
                    mins = np.min(cover_count[balls[selected]], axis=1)
                    removable_mask = mins > 1
                    if not np.any(removable_mask):
                        break
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    changed = True
                    
            return selected, cover_count

        selected, cover_count = try_swaps(selected, cover_count)
        
        # Try 2-for-1 swaps
        def try_2_for_1(selected, cover_count, max_iters=100):
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                if len(selected) < 2:
                    break
                
                # Prune first
                mins = np.min(cover_count[balls[selected]], axis=1)
                removable_mask = mins > 1
                if np.any(removable_mask):
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    continue
                
                idx1, idx2 = rng.choice(len(selected), 2, replace=False)
                c1 = selected[idx1]
                c2 = selected[idx2]
                
                words_c1 = balls[c1]
                words_c2 = balls[c2]
                
                unique_c1 = words_c1[cover_count[words_c1] == 1]
                unique_c2 = words_c2[cover_count[words_c2] == 1]
                
                # Shared words with count 2
                # Use bitsets to find intersection of words?
                # balls are arrays of words.
                # We need words w such that w in B(c1) AND w in B(c2) AND cover_count[w]==2.
                # This is the intersection of the two ball sets.
                # We can compute this by checking cover_count?
                # If w is in both, and count is 2, it must be covered by c1 and c2 only.
                
                # Efficient intersection of two small lists (size 17)
                # Convert to set for intersection
                set_c1 = set(words_c1.tolist())
                set_c2 = set(words_c2.tolist())
                shared = list(set_c1 & set_c2)
                if shared:
                    shared_arr = np.array(shared, dtype=np.int32)
                    shared_unique = shared_arr[cover_count[shared_arr] == 2]
                else:
                    shared_unique = np.array([], dtype=np.int32)
                
                all_to_cover = np.concatenate([unique_c1, unique_c2, shared_unique])
                
                if len(all_to_cover) == 0:
                    continue
                
                # Find c3 covering all_to_cover
                candidates_bits = word_to_codewords_bits[all_to_cover[0]]
                for w in all_to_cover[1:]:
                    candidates_bits &= word_to_codewords_bits[w]
                
                # Exclude c1, c2
                candidates_bits &= ~(1 << c1)
                candidates_bits &= ~(1 << c2)
                
                if candidates_bits == 0:
                    continue
                
                cand_list = []
                temp_bits = candidates_bits
                while temp_bits:
                    lsb = temp_bits & -temp_bits
                    cand_list.append(lsb.bit_length() - 1)
                    temp_bits ^= lsb
                    
                if not cand_list:
                    continue
                
                c3 = cand_list[rng.integers(len(cand_list))]
                
                # Perform swap
                cover_count[balls[c1]] -= 1
                cover_count[balls[c2]] -= 1
                cover_count[balls[c3]] += 1
                
                new_selected = []
                for i, c in enumerate(selected):
                    if i != idx1 and i != idx2:
                        new_selected.append(c)
                new_selected.append(c3)
                selected = np.array(new_selected, dtype=np.int32)
                
                # Prune
                changed = True
                while changed and time.time() < deadline - 5.0:
                    mins = np.min(cover_count[balls[selected]], axis=1)
                    removable_mask = mins > 1
                    if not np.any(removable_mask):
                        break
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    changed = True
                    
            return selected, cover_count

        selected, cover_count = try_2_for_1(selected, cover_count)

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break

    if best_set is None:
        best_set = np.arange(total, dtype=np.int32)
        
    result = []
    for w in best_set:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
