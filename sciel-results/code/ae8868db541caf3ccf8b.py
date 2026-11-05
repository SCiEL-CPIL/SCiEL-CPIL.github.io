import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    total_words = q ** n
    
    # 1. Precompute neighbors as bitsets for fast coverage checks
    #    Each word's ball is a Python integer (bitset)
    #    6561 words -> 6561 bits -> ~820 bytes per bitset. 6561*820 ~ 5MB. OK.
    neighbors_bitset = [0] * total_words
    
    # Helper to convert index to digits
    def idx_to_digits(i):
        digits = [0] * n
        for j in range(n):
            digits[j] = i % q
            i //= q
        return digits
    
    # Helper to convert digits to index
    def digits_to_idx(digits):
        idx = 0
        for j in range(n):
            idx = idx * q + digits[j]
        return idx

    # Precompute neighbors
    # Optimization: We can generate neighbors more efficiently
    # For radius 1:
    #   1 + n*(q-1) neighbors
    # For general radius, it's (1 + (q-1))^r ...
    # Given n=8, q=3, radius=1 is likely the target (K_3(8,1)).
    # If radius > 1, this precomputation might be slow, but let's try.
    
    # Let's generate all words' neighbors
    # To speed up, we can precompute the "difference vectors" or just iterate.
    
    # Efficient neighbor generation:
    # For each word w, its neighbors are w plus all vectors v with Hamming weight <= radius and v != 0.
    
    # Precompute all offset vectors with weight <= radius
    offsets = []
    # Weight 0
    offsets.append([0]*n)
    # Weight 1..radius
    for r in range(1, radius + 1):
        for pos_comb in itertools.combinations(range(n), r):
            # Non-zero values for these positions
            for vals in itertools.product(range(1, q), repeat=r):
                offset = [0]*n
                for k, pos in enumerate(pos_comb):
                    offset[pos] = vals[k]
                offsets.append(offset)
    
    # Now compute bitsets
    for i in range(total_words):
        bits = 1 << i
        d = idx_to_digits(i)
        for off in offsets:
            if off == [0]*n: continue
            new_d = [(d[j] + off[j]) % q for j in range(n)]
            j = digits_to_idx(new_d)
            bits |= (1 << j)
        neighbors_bitset[i] = bits
        
    # 2. Greedy Initialization
    #    Start with empty code, repeatedly add word covering most uncovered
    uncovered_mask = (1 << total_words) - 1
    code = []
    code_set = set()
    
    while uncovered_mask and time.time() < deadline - 30.0:
        best_idx = -1
        max_cov = 0
        # To speed up, we can sample or check all. 6561 is small enough for linear scan if optimized.
        # But checking 6561 bitset intersections is fast in Python? 
        # popcount of (bitset & mask)
        # Python int bit_count() is fast.
        
        # Optimization: Instead of checking all 6561, we can just pick a random one from uncovered?
        # No, greedy is better.
        # Let's iterate.
        for i in range(total_words):
            if i in code_set:
                continue
            cov = (neighbors_bitset[i] & uncovered_mask).bit_count()
            if cov > max_cov:
                max_cov = cov
                best_idx = i
            if max_cov == 0:
                break
        if best_idx == -1 or max_cov == 0:
            break
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered_mask &= ~neighbors_bitset[best_idx]
        
    if not code:
        # Fallback: Tetracode product
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_code = code[:]
    best_size = len(best_code)
    
    # 3. Local Search: 2-Opt (Remove one, Add one)
    #    We maintain the code as a list.
    #    We check if removing a codeword makes the code invalid.
    #    If it does, we try to find a replacement that restores validity and allows removal.
    #    Actually, standard ILS:
    #    1. Remove a random codeword.
    #    2. If invalid, add a random codeword to fix it (or the one that covers most newly uncovered).
    #    3. If valid, keep.
    #    4. If size decreases, accept.
    #    5. If size same, accept with probability (simulated annealing) or just accept if it helps future moves?
    #       For minimization, we only accept if size < best_size.
    #       But we need to escape local minima.
    #       Strategy: Accept moves that keep size same but change the set (random walk) to explore.
    #       Keep track of best found.
    
    current_code = best_code[:]
    current_set = set(current_code)
    current_size = len(current_code)
    
    # Precompute for speed: 
    # To check if removing 'x' is safe, we need to know if any word is ONLY covered by 'x'.
    # This requires knowing the coverage count of each word by the current code.
    # Maintaining counts:
    # counts[w] = number of codewords in current_code whose ball contains w.
    
    # Initialize counts
    counts = np.zeros(total_words, dtype=np.int32)
    for cw in current_code:
        # Add 1 to all neighbors of cw
        # This is slow if done naively.
        # Let's use the bitset? No, bitset doesn't give counts.
        # We can iterate through neighbors.
        # Radius 1: 1 + 8*2 = 17 neighbors. Very fast.
        d_cw = idx_to_digits(cw)
        counts[cw] += 1
        for r in range(1, radius + 1):
            for pos_comb in itertools.combinations(range(n), r):
                for vals in itertools.product(range(1, q), repeat=r):
                    new_d = d_cw[:]
                    for k, pos in enumerate(pos_comb):
                        new_d[pos] = (new_d[pos] + vals[k]) % q
                    j = digits_to_idx(new_d)
                    counts[j] += 1

    # Helper to get neighbors list (indices) for a word
    # We already have bitsets. We can extract indices if needed, but iterating is better for counts.
    # Let's create a list of neighbor indices for each word for fast iteration.
    neighbors_list = [[] for _ in range(total_words)]
    for i in range(total_words):
        bits = neighbors_bitset[i]
        # Extract set bits
        # This is expensive to do for all 6561 words initially?
        # 6561 * 17 bits set.
        # Let's just generate them again, it's cheap.
        d_i = idx_to_digits(i)
        neighbors_list[i].append(i)
        for r in range(1, radius + 1):
            for pos_comb in itertools.combinations(range(n), r):
                for vals in itertools.product(range(1, q), repeat=r):
                    new_d = d_i[:]
                    for k, pos in enumerate(pos_comb):
                        new_d[pos] = (new_d[pos] + vals[k]) % q
                    j = digits_to_idx(new_d)
                    neighbors_list[i].append(j)
                    
    # ILS Loop
    # We try to remove codewords.
    # If we remove a codeword and some words become uncovered (count 0), we must add a new codeword to cover them.
    # To minimize size, we want to add a codeword that covers as many of the "lost" words as possible, 
    # and ideally also allows removing another codeword? No, just restore validity.
    # If we can remove 1 and add 1, size stays same. 
    # If we can remove 1 and add 0 (because no words were uniquely covered), size decreases.
    
    # We will try:
    # 1. Pick a random codeword to remove.
    # 2. Check if any word has count 1 (and that codeword is the coverer).
    #    If none, remove it. Update counts. Accept.
    # 3. If some words become uncovered, find a replacement codeword that covers the most of them.
    #    Add it. Update counts.
    #    If size didn't decrease, we might still accept to explore (random walk) if we are stuck?
    #    Let's stick to: Accept if size decreases. If size same, accept with small probability or just reject to keep best?
    #    Better: Keep a "current" state that can be worse, but track "best".
    #    Let's do: Accept if size < best_size. If size == best_size, accept with 50% chance (to move around).
    #    If size > best_size, reject.
    
    # Actually, a simpler and effective strategy for covering codes:
    # "Shake" the code: Remove a random codeword. If invalid, add the best candidate to fix it.
    # Repeat.
    
    best_size = len(current_code)
    best_code = current_code[:]
    
    iterations = 0
    while time.time() < deadline - 5.0:
        iterations += 1
        # Pick a random codeword to remove
        if not current_code:
            break
        rem_idx = random.randint(0, len(current_code) - 1)
        rem_cw = current_code[rem_idx]
        
        # Tentatively remove
        # Check if any word becomes uncovered
        # Words with count[rem_cw's neighbors] == 1 are at risk.
        # Specifically, if count[w] == 1 and rem_cw covers w, then w becomes uncovered.
        
        # Let's find the set of words that would become uncovered.
        # We need to know which words are covered ONLY by rem_cw.
        # This is equivalent to: for w in neighbors(rem_cw), if counts[w] == 1, then w is lost.
        
        lost_words = []
        for w in neighbors_list[rem_cw]:
            if counts[w] == 1:
                lost_words.append(w)
                
        if not lost_words:
            # Safe to remove
            current_code.pop(rem_idx)
            current_set.discard(rem_cw)
            # Update counts
            for w in neighbors_list[rem_cw]:
                counts[w] -= 1
            current_size = len(current_code)
            
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
                
        else:
            # Need to add a codeword to cover lost_words
            # Find candidate that covers most lost_words
            # To speed up, we can just check candidates from a pool or all words.
            # Checking all 6561 words is fast if we use bitsets for lost_words?
            # lost_mask = 0
            # for w in lost_words: lost_mask |= 1 << w
            # best_cov = 0
            # for cand in range(total_words):
            #    if cand in current_set: continue
            #    cov = (neighbors_bitset[cand] & lost_mask).bit_count()
            #    ...
            
            lost_mask = 0
            for w in lost_words:
                lost_mask |= (1 << w)
                
            best_cand = -1
            best_cov = 0
            # Random subset of candidates to save time? Or all?
            # 6561 checks * bit_count is very fast.
            for cand in range(total_words):
                if cand in current_set:
                    continue
                cov = (neighbors_bitset[cand] & lost_mask).bit_count()
                if cov > best_cov:
                    best_cov = cov
                    best_cand = cand
                    
            if best_cand == -1:
                continue
                
            # Add best_cand
            # But first, do we remove rem_cw?
            # If we add best_cand, does it cover all lost_words?
            # If best_cov == len(lost_words), then yes.
            # If not, we can't just swap. We'd have to add without removing, increasing size.
            # We only want to swap if it fixes the loss.
            if best_cov == len(lost_words):
                # Swap: remove rem_cw, add best_cand
                current_code[rem_idx] = best_cand
                current_set.discard(rem_cw)
                current_set.add(best_cand)
                
                # Update counts
                # Decrement for rem_cw
                for w in neighbors_list[rem_cw]:
                    counts[w] -= 1
                # Increment for best_cand
                for w in neighbors_list[best_cand]:
                    counts[w] += 1
                    
                # Size is same. 
                # Do we accept?
                # If we accept, we move to a new state of same size. This helps escape local minima.
                # Let's accept always in this mode to explore, but only update best if size drops (which it doesn't here).
                # Wait, if size doesn't drop, we don't update best.
                # But we are moving the "current" state.
                # This is a random walk on the graph of valid codes of fixed size.
                # We should occasionally try to reduce size.
                # The "remove if safe" step handles reduction.
                # So this swap allows us to reach a state where a different codeword might be removable.
                
                # To avoid infinite loops in same size, we can limit consecutive swaps?
                # Or just let it run.
                
            else:
                # Cannot fix with single swap. 
                # Try adding best_cand WITHOUT removing rem_cw?
                # This increases size. We only do this if we are stuck?
                # Let's stick to swaps for now.
                pass

    return [idx_to_digits(i) for i in best_code]
