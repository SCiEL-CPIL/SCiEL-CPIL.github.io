import numpy as np
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 18
    MIN_DIST = 6
    
    # Start with Reed-Muller [16,5,8] code extended by 2 zero bits -> 32 words, dist 8
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
    
    best_code = list(base_words)
    best_len = len(best_code)
    
    # Precompute popcount lookup for 16-bit chunks
    popcount_table = np.zeros(65536, dtype=np.uint8)
    for i in range(65536):
        popcount_table[i] = bin(i).count('1')
    
    def popcount_fast(arr):
        """Fast popcount for uint32 array using table lookup on 16-bit halves."""
        low = (arr & 0xFFFF).astype(np.uint32)
        high = (arr >> 16).astype(np.uint32)
        return (popcount_table[low] + popcount_table[high]).astype(np.uint8)
    
    def greedy_grow(code, code_set, rng, time_limit):
        """Grow code using greedy addition with random candidate order."""
        code = list(code)
        code_set = set(code)
        code_len = len(code)
        
        # Pre-allocate
        max_code_size = 2**18
        code_arr = np.zeros(max_code_size, dtype=np.uint32)
        code_arr[:code_len] = np.array(code, dtype=np.uint32)
        
        # Generate shuffled candidates
        all_words = np.arange(2**18, dtype=np.uint32)
        rng.shuffle(all_words)
        
        added_count = 0
        start_time = time.time()
        
        for i in range(len(all_words)):
            if time.time() > time_limit:
                break
            
            candidate = all_words[i]
            candidate_int = int(candidate)
            if candidate_int in code_set:
                continue
            
            # Check distance to all existing code words
            xors = code_arr[:code_len] ^ candidate
            popcounts = popcount_fast(xors)
            
            min_d = np.min(popcounts)
            
            if min_d >= MIN_DIST:
                code.append(candidate_int)
                code_set.add(candidate_int)
                code_arr[code_len] = candidate_int
                code_len += 1
                added_count += 1
        
        return code, code_set, code_len
    
    # Do multiple restarts with different seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(seed * 12345 + 67890)
        
        # Try growing from the base code
        code, code_set, code_len = greedy_grow(
            base_words, set(base_words), rng, deadline
        )
        
        if len(code) > best_len:
            best_code = code
            best_len = len(code)
    
    # Try a second phase: local search to improve the best code
    # Try to swap out words that block many additions
    if time.time() < deadline - 5:
        rng = np.random.default_rng(99999)
        best_code_set = set(best_code)
        
        # Try removing one word and re-adding greedily
        for remove_idx in range(len(best_code)):
            if time.time() > deadline:
                break
            
            test_code = best_code[:remove_idx] + best_code[remove_idx+1:]
            test_set = set(test_code)
            test_len = len(test_code)
            
            code_arr = np.zeros(2**18, dtype=np.uint32)
            code_arr[:test_len] = np.array(test_code, dtype=np.uint32)
            
            # Try to add back and more
            all_words = np.arange(2**18, dtype=np.uint32)
            rng.shuffle(all_words)
            
            improved = False
            for candidate in all_words:
                if time.time() > deadline:
                    break
                
                cand_int = int(candidate)
                if cand_int in test_set:
                    continue
                
                xors = code_arr[:test_len] ^ candidate
                popcounts = popcount_fast(xors)
                
                if np.min(popcounts) >= MIN_DIST:
                    test_code.append(cand_int)
                    test_set.add(cand_int)
                    code_arr[test_len] = cand_int
                    test_len += 1
                    improved = True
            
            if len(test_code) > best_len:
                best_code = test_code
                best_len = len(test_code)
    
    return best_code
