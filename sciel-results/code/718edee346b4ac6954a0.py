import time
import itertools
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds limit to be safe
    
    # Helper to convert word to int
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
        
    # Helper to convert int to word
    def int_to_word(w_int):
        w = []
        tmp = w_int
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        return list(reversed(w))

    num_words = q ** n
    all_words_ints = list(range(num_words))
    
    # Precompute balls for all possible words as bitsets (Python integers)
    # bit i is 1 if word i is in the ball
    ball_cache = [0] * num_words
    for w_int in all_words_ints:
        w = []
        tmp = w_int
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        w = tuple(reversed(w))
        
        ball = 1 << w_int
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        new_w = list(w)
                        new_w[i] = val
                        new_int = word_to_int(tuple(new_w))
                        ball |= (1 << new_int)
        # For radius > 1, this is incomplete. 
        # But for K_3(6,1), radius is 1.
        # If radius > 1 is passed, we need BFS. 
        # Given the task context, we assume radius=1. 
        # To be safe, if radius > 1, we fallback to a simple construction.
        if radius > 1:
             # Simple BFS for general radius
             visited = {w_int}
             queue = [(w_int, 0)]
             while queue:
                 cur, dist = queue.pop(0)
                 if dist == radius:
                     continue
                 cur_w = int_to_word(cur)
                 for i in range(n):
                     for val in range(q):
                         if val != cur_w[i]:
                             new_w = cur_w[:]
                             new_w[i] = val
                             new_int = word_to_int(tuple(new_w))
                             if new_int not in visited:
                                 visited.add(new_int)
                                 queue.append((new_int, dist + 1))
             ball = 0
             for v in visited:
                 ball |= (1 << v)
        ball_cache[w_int] = ball

    # Greedy Construction
    def greedy_construct():
        uncovered = (1 << num_words) - 1
        code = []
        while uncovered:
            best_c = -1
            best_covered = 0
            # To speed up, we can sample or iterate all. 729 is small.
            for c in range(num_words):
                covered = ball_cache[c] & uncovered
                count = bin(covered).count('1')
                if count > best_covered:
                    best_covered = count
                    best_c = c
                    if count == 0:
                        break # Should not happen if uncovered != 0
            code.append(best_c)
            uncovered &= ~ball_cache[best_c]
        return code

    # Greedy Removal
    def greedy_remove(code):
        code_set = set(code)
        # Precompute union of balls for all but one? 
        # Simpler: for each c, check if any word in ball(c) is ONLY covered by c.
        # We can maintain cover counts.
        cover_count = [0] * num_words
        for c in code_set:
            b = ball_cache[c]
            # Iterate bits
            tmp = b
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                tmp ^= lsb
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for c in list(code_set):
                b = ball_cache[c]
                # Check if any word in b has cover_count == 1
                removable = True
                tmp = b
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        removable = False
                        break
                    tmp ^= lsb
                
                if removable:
                    code_set.remove(c)
                    # Update cover counts
                    tmp = b
                    while tmp:
                        lsb = tmp & -tmp
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        tmp ^= lsb
                    changed = True
        return list(code_set)

    # Local Search (Swaps)
    def local_search(code):
        code_set = set(code)
        cover_count = [0] * num_words
        for c in code_set:
            b = ball_cache[c]
            tmp = b
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                tmp ^= lsb
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to remove a codeword if possible (redundant)
            # This is handled by greedy_remove, but let's integrate it or call it.
            # Actually, let's just do swaps.
            # Pick a random codeword
            c = random.choice(list(code_set))
            b_c = ball_cache[c]
            
            # Find words covered ONLY by c
            critical_words = []
            tmp = b_c
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    critical_words.append(idx)
                tmp ^= lsb
            
            if not critical_words:
                # Can remove c
                code_set.remove(c)
                tmp = b_c
                while tmp:
                    lsb = tmp & -tmp
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    tmp ^= lsb
                improved = True
                # Call greedy_remove to clean up any new redundancies
                new_code = greedy_remove(list(code_set))
                code_set = set(new_code)
                # Recalculate cover counts
                cover_count = [0] * num_words
                for c_new in code_set:
                    b_new = ball_cache[c_new]
                    tmp_new = b_new
                    while tmp_new:
                        lsb_new = tmp_new & -tmp_new
                        idx_new = lsb_new.bit_length() - 1
                        cover_count[idx_new] += 1
                        tmp_new ^= lsb_new
                continue
            
            # Find a non-codeword c' that covers all critical_words
            # critical_words is a set of indices. We need a ball that contains all of them.
            # We can precompute a bitset for critical_words
            crit_mask = 0
            for idx in critical_words:
                crit_mask |= (1 << idx)
            
            found_swap = False
            # Iterate through all potential c'
            # We can skip c itself
            for c_prime in range(num_words):
                if c_prime in code_set:
                    continue
                if (ball_cache[c_prime] & crit_mask) == crit_mask:
                    # Valid swap
                    # Remove c
                    code_set.remove(c)
                    tmp = b_c
                    while tmp:
                        lsb = tmp & -tmp
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        tmp ^= lsb
                    
                    # Add c_prime
                    code_set.add(c_prime)
                    b_cp = ball_cache[c_prime]
                    tmp = b_cp
                    while tmp:
                        lsb = tmp & -tmp
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        tmp ^= lsb
                    
                    improved = True
                    found_swap = True
                    # Call greedy_remove to clean up
                    new_code = greedy_remove(list(code_set))
                    code_set = set(new_code)
                    # Recalculate cover counts
                    cover_count = [0] * num_words
                    for c_new in code_set:
                        b_new = ball_cache[c_new]
                        tmp_new = b_new
                        while tmp_new:
                            lsb_new = tmp_new & -tmp_new
                            idx_new = lsb_new.bit_length() - 1
                            cover_count[idx_new] += 1
                            tmp_new ^= lsb_new
                    break
            
            if not found_swap:
                # Try next random codeword
                continue
        
        return list(code_set)

    best_code = None
    best_size = float('inf')
    
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # 1. Greedy Construct
        code = greedy_construct()
        
        # 2. Greedy Remove
        code = greedy_remove(code)
        
        # 3. Local Search
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            if best_size == 73:
                # We found the known optimum. We can stop or try to find 72 (unlikely).
                # Let's do a few more restarts just in case, but we can break if time is tight.
                if time.time() > deadline - 20:
                    break

    if best_code is None:
        # Fallback to direct sum
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            seed_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
            best_code = [word_to_int(tuple(c)) for c in seed_codewords]
        else:
            best_code = list(range(num_words))

    return [int_to_word(c_int) for c_int in best_code]
