import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (balls of radius 1) as bitsets
    # ball[i] is an integer where bit j is 1 if dist(words[i], words[j]) <= radius
    balls = [0] * N
    for i, w in enumerate(words):
        mask = 0
        w_list = list(w)
        # Self
        mask |= (1 << i)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                mask |= (1 << idx)
                w_list[pos] = orig_val
        balls[i] = mask
        
    UNIVERSE = (1 << N) - 1

    # 3. Greedy Set Cover
    def greedy_cover(uncovered_mask, current_code, current_counts):
        # current_code: set of indices
        # current_counts: list of counts
        # uncovered_mask: int bitset of uncovered points
        code = set(current_code)
        counts = current_counts[:]
        
        while uncovered_mask:
            if time.time() > deadline - 5:
                break
            
            best_idx = -1
            max_gain = 0
            
            # Find word covering most uncovered points
            for i in range(N):
                # Gain is number of uncovered points covered by i
                gain = (uncovered_mask & balls[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            if max_gain == 0:
                break
                
            code.add(best_idx)
            # Update counts and uncovered
            # We need to know which points were newly covered
            new_covered = uncovered_mask & balls[best_idx]
            uncovered_mask &= ~new_covered
            
            # Update counts for all points in balls[best_idx]
            # This is O(N) per addition if we iterate bits, but we can just 
            # increment counts for all j in balls[best_idx]
            # Since N is small (2187), iterating set bits is fine.
            m = balls[best_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
                
        return code, counts, uncovered_mask

    # Initial Greedy
    initial_uncovered = UNIVERSE
    initial_counts = [0] * N
    best_code, best_counts, _ = greedy_cover(initial_uncovered, set(), initial_counts)
    
    # Helper to check validity and get counts from a code set
    def get_counts(code_set):
        counts = [0] * N
        for c in code_set:
            m = balls[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    # Initial polish (deletion)
    def polish(code_set, counts):
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for c in list(code_set):
                # Check if c is redundant
                # c is redundant if for all j in balls[c], counts[j] > 1
                redundant = True
                m = balls[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    code_set.remove(c)
                    # Decrement counts
                    m = balls[c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    changed = True
    
    polish(best_code, best_counts)
    best_size = len(best_code)
    
    # 4. Kick-Repair Local Search
    # We will try to improve best_code
    # State: code_set, counts
    
    # To speed up repair, we can precompute list of indices for each ball?
    # No, bit iteration is fast enough.
    
    kick_sizes = [1, 2, 3, 4, 5]
    
    while time.time() < deadline - 5:
        # Select a kick size
        k = random.choice(kick_sizes)
        
        # Copy current best
        curr_code = set(best_code)
        curr_counts = best_counts[:]
        
        # 1. Kick: Remove k random codewords
        if len(curr_code) <= k:
            continue
            
        # Randomly select k codewords to remove
        # Converting to list for random selection
        code_list = list(curr_code)
        random.shuffle(code_list)
        to_remove = code_list[:k]
        
        for c in to_remove:
            curr_code.remove(c)
            # Decrement counts
            m = balls[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                curr_counts[idx] -= 1
                m ^= lsb
                
        # 2. Repair: Cover the holes
        # Find uncovered points
        uncovered = 0
        for j in range(N):
            if curr_counts[j] == 0:
                uncovered |= (1 << j)
                
        # Greedy repair
        while uncovered:
            if time.time() > deadline - 2:
                break
            
            best_idx = -1
            max_gain = 0
            
            for i in range(N):
                gain = (uncovered & balls[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            if max_gain == 0:
                break # Should not happen if valid code existed before kick
                
            curr_code.add(best_idx)
            new_covered = uncovered & balls[best_idx]
            uncovered &= ~new_covered
            
            # Update counts
            m = balls[best_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                curr_counts[idx] += 1
                m ^= lsb
                
        # 3. Polish: Remove redundant codewords
        polish(curr_code, curr_counts)
        
        # 4. Accept if better
        if len(curr_code) < best_size:
            best_code = curr_code
            best_counts = curr_counts
            best_size = len(best_code)
            
        # If we hit the lower bound or a very good size, we might want to 
        # intensify search or break? No, keep searching.
        
    return [list(words[i]) for i in best_code]
