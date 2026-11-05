import time
import random
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def is_valid_fast(seq):
        """Fast validity check using prefix sums"""
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        
        sums = set()
        for i in range(L):
            pi = P[i]
            for j in range(i + 1, L + 1):
                s = P[j] - pi
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_build(order):
        """Greedy construction following a given order of candidates"""
        values = []
        sums = set()
        for x in order:
            suffix_sums = []
            current_sum = 0
            for v in reversed(values):
                current_sum += v
                suffix_sums.append(current_sum)
            
            potential_sums = [x] + [x + s for s in suffix_sums]
            
            valid = True
            for ps in potential_sums:
                if ps in sums:
                    valid = False
                    break
            
            if valid:
                values.append(x)
                for ps in potential_sums:
                    sums.add(ps)
        
        return values

    # Try multiple greedy orderings
    best_seq = []
    best_len = 0
    
    # Ordering 1: natural ascending
    seq1 = greedy_build(list(range(1, 257)))
    if len(seq1) > best_len:
        best_len = len(seq1)
        best_seq = seq1
    
    # Ordering 2: descending
    seq2 = greedy_build(list(range(256, 0, -1)))
    if len(seq2) > best_len:
        best_len = len(seq2)
        best_seq = seq2
    
    # Ordering 3: interleaved (small, large, small, large)
    interleaved = []
    for i in range(128):
        interleaved.append(i + 1)
        interleaved.append(256 - i)
    seq3 = greedy_build(interleaved)
    if len(seq3) > best_len:
        best_len = len(seq3)
        best_seq = seq3
    
    # Ordering 4: powers of 2 first, then others
    powers = [2**i for i in range(1, 9) if 2**i <= 256]
    others = [x for x in range(1, 257) if x not in powers]
    ordering4 = powers + others
    seq4 = greedy_build(ordering4)
    if len(seq4) > best_len:
        best_len = len(seq4)
        best_seq = seq4
    
    # Ordering 5: Fibonacci-like
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= 256:
        fibs.append(fibs[-1] + fibs[-2])
    others5 = [x for x in range(1, 257) if x not in fibs]
    ordering5 = fibs + others5
    seq5 = greedy_build(ordering5)
    if len(seq5) > best_len:
        best_len = len(seq5)
        best_seq = seq5
    
    # Random orderings
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        random_order = list(range(1, 257))
        random.shuffle(random_order)
        seq_r = greedy_build(random_order)
        if len(seq_r) > best_len:
            best_len = len(seq_r)
            best_seq = seq_r
            
    def local_search_improve(seq):
        if not seq:
            return seq
        
        # Build initial sums set
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            pi = P[i]
            for j in range(i + 1, L + 1):
                sums.add(P[j] - pi)
        
        current_seq = list(seq)
        current_sums = sums
        current_set = set(current_seq)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try adding elements
            for x in range(1, 257):
                if x in current_set:
                    continue
                
                # Find insertion position
                pos = bisect.bisect_left(current_seq, x)
                
                # Calculate new sums involving x
                # Left part: current_seq[0:pos]
                # Right part: current_seq[pos:]
                
                new_sums = []
                
                # Sums ending at x (x + suffixes of left part)
                sum_left = 0
                for i in range(pos - 1, -1, -1):
                    sum_left += current_seq[i]
                    new_sums.append(sum_left + x)
                new_sums.append(x) # x itself
                
                # Sums starting at x (x + prefixes of right part)
                sum_right = 0
                for i in range(pos, len(current_seq)):
                    sum_right += current_seq[i]
                    if i > pos: # Avoid duplicate x
                        new_sums.append(x + sum_right)
                
                # Sums spanning x (suffix_left + x + prefix_right)
                # l > 0, r > 0
                left_suffixes = []
                s = 0
                for i in range(pos - 1, -1, -1):
                    s += current_seq[i]
                    left_suffixes.append(s)
                    
                right_prefixes = []
                s = 0
                for i in range(pos, len(current_seq)):
                    s += current_seq[i]
                    right_prefixes.append(s)
                
                for l in left_suffixes:
                    for r in right_prefixes:
                        new_sums.append(l + x + r)
                
                # Check for collisions
                valid = True
                # 1. Check against existing sums
                for ns in new_sums:
                    if ns in current_sums:
                        valid = False
                        break
                
                # 2. Check internal collisions in new_sums
                if valid:
                    if len(new_sums) != len(set(new_sums)):
                        valid = False
                
                if valid:
                    # Accept insertion
                    current_seq.insert(pos, x)
                    current_set.add(x)
                    current_sums.update(new_sums)
                    improved = True
                    # Break to restart the loop with new state
                    break
            
        return current_seq

    # Run local search on the best found so far
    final_seq = local_search_improve(best_seq)
    
    if len(final_seq) > best_len:
        best_seq = final_seq
        best_len = len(final_seq)
        
    return sorted(best_seq)
