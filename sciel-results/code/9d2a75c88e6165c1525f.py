import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    if n != 6 or q != 3 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy for general case
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        # Precompute coverage
        coverage = []
        for w in words:
            cov = set([w])
            for i in range(n):
                for delta in range(-radius, radius+1):
                    if delta == 0: continue
                    new_w = list(w)
                    new_w[i] = (new_w[i] + delta) % q
                    if all(0 <= x < q for x in new_w):
                        cov.add(tuple(new_w))
            coverage.append(cov)
            
        uncovered = set(words)
        selected = []
        
        while uncovered:
            best_w = None
            best_count = -1
            for i, w in enumerate(words):
                if w in [tuple(s) for s in selected]: continue
                count = len(uncovered & coverage[i])
                if count > best_count:
                    best_count = count
                    best_w = i
            if best_count == 0: break
            selected.append(words[best_w])
            uncovered -= coverage[best_w]
            
        return [list(w) for w in selected]

    # Specific optimized path for q=3, n=6, radius=1
    total_words = 3 ** 6
    words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute coverage bitsets (using Python integers)
    # Bit i is 1 if word i is covered
    coverage = [0] * total_words
    for i, w in enumerate(words):
        bits = 1 << i
        for pos in range(6):
            for delta in [-1, 1]:
                nw = list(w)
                nw[pos] = (nw[pos] + delta) % 3
                idx = word_to_idx[tuple(nw)]
                bits |= (1 << idx)
        coverage[i] = bits
        
    full_mask = (1 << total_words) - 1
    
    # Greedy Phase
    uncovered = full_mask
    selected = []
    selected_idx = set()
    
    while uncovered:
        best_idx = -1
        best_count = -1
        # Optimization: only check words not already selected?
        # Actually, adding a selected word again is useless.
        # We iterate through all words, but skip selected ones.
        for i in range(total_words):
            if i in selected_idx:
                continue
            count = bin(uncovered & coverage[i]).count('1')
            if count > best_count:
                best_count = count
                best_idx = i
                if count == bin(uncovered).count('1'):
                    break
        if best_count == 0:
            break
        selected.append(best_idx)
        selected_idx.add(best_idx)
        uncovered &= ~coverage[best_idx]
        
    # Local Search Phase
    # cover_count[w] = number of selected codewords covering w
    cover_count = [0] * total_words
    for idx in selected:
        bits = coverage[idx]
        # Iterate set bits
        b = bits
        while b:
            lsb = b & -b
            pos = lsb.bit_length() - 1
            cover_count[pos] += 1
            b ^= lsb
            
    def is_valid_removal(idx):
        bits = coverage[idx]
        b = bits
        while b:
            lsb = b & -b
            pos = lsb.bit_length() - 1
            if cover_count[pos] <= 1:
                return False
            b ^= lsb
        return True

    def do_removal(idx):
        bits = coverage[idx]
        b = bits
        while b:
            lsb = b & -b
            pos = lsb.bit_length() - 1
            cover_count[pos] -= 1
            b ^= lsb
        selected.remove(idx)
        selected_idx.remove(idx)

    def do_addition(idx):
        bits = coverage[idx]
        b = bits
        while b:
            lsb = b & -b
            pos = lsb.bit_length() - 1
            cover_count[pos] += 1
            b ^= lsb
        selected.append(idx)
        selected_idx.add(idx)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove
        for idx in list(selected):
            if is_valid_removal(idx):
                do_removal(idx)
                improved = True
                # Restart removal loop immediately as state changed
                break
        
        if time.time() > deadline: break

        # 2. Try to swap (Add one, remove one)
        # To speed up, we can just try adding a random unselected word, then see if we can remove something.
        # Or iterate all pairs. 729 * 73 is ~53k pairs. Very fast.
        for add_idx in range(total_words):
            if add_idx in selected_idx:
                continue
            
            # Check if adding add_idx allows removal of any existing idx
            # We need to simulate:
            # 1. Add add_idx (increment counts)
            # 2. For each existing idx, check if removal is valid
            # 3. If yes, commit swap.
            
            # To avoid expensive rollback, we can check validity analytically?
            # Removal of `rem_idx` is valid if for all w in coverage[rem_idx], 
            # (cover_count[w] - 1 + (1 if w in coverage[add_idx] else 0)) > 0
            # i.e., cover_count[w] + (1 if w in coverage[add_idx] else 0) > 1
            
            # Let's just simulate it for the most promising candidates or all.
            # Since N is small, we can just try all.
            
            # Optimization: Only try swaps if adding add_idx covers some "critical" words?
            # For now, brute force swaps.
            
            # We can break early if we find a valid swap.
            for rem_idx in list(selected):
                valid = True
                bits_rem = coverage[rem_idx]
                b = bits_rem
                while b:
                    lsb = b & -b
                    pos = lsb.bit_length() - 1
                    # New count for pos if we remove rem_idx and add add_idx
                    new_count = cover_count[pos] - 1
                    if coverage[add_idx] & (1 << pos):
                        new_count += 1
                    if new_count == 0:
                        valid = False
                        break
                    b ^= lsb
                if valid:
                    # Perform swap
                    do_addition(add_idx)
                    do_removal(rem_idx)
                    improved = True
                    break
            if improved:
                break
                
    return [list(words[idx]) for idx in selected]
