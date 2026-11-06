import numpy as np
import time

def construct():
    deadline = time.time() + 85
    
    N = 18
    MIN_DIST = 6
    
    # Reed-Muller seed: 32 words with distance 8
    base_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base_words.append(word)
    
    code = list(base_words)
    code_set = set(code)
    
    # Precompute popcount for all 18-bit values
    popcount_table = np.zeros(2**N, dtype=np.int8)
    for i in range(1, 2**N):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    def fast_greedy_add(code_list, code_set_local, all_candidates, max_time):
        code_arr = np.array(code_list, dtype=np.uint32)
        n_code = len(code_list)
        
        chunk_size = 4096
        
        for start in range(0, len(all_candidates), chunk_size):
            if time.time() > max_time:
                break
            
            chunk = all_candidates[start:start+chunk_size]
            chunk_filtered = [c for c in chunk if c not in code_set_local]
            if not chunk_filtered:
                continue
            
            cand_arr = np.array(chunk_filtered, dtype=np.uint32)
            min_dists = np.full(len(cand_arr), 100, dtype=np.int32)
            
            code_chunk_size = max(1, n_code // 4)
            for ci in range(0, n_code, code_chunk_size):
                code_chunk = code_arr[ci:ci+code_chunk_size]
                xors = cand_arr[:, None] ^ code_chunk[None, :]
                pc = popcount_table[xors]
                min_dists = np.minimum(min_dists, pc.min(axis=1))
                if (min_dists < MIN_DIST).all():
                    break
            
            for i in range(len(cand_arr)):
                if min_dists[i] >= MIN_DIST:
                    w = int(cand_arr[i])
                    code_list.append(w)
                    code_set_local.add(w)
                    if len(code_list) % 64 == 0:
                        code_arr = np.array(code_list, dtype=np.uint32)
                        n_code = len(code_list)
        
        return len(code_list)
    
    rng = np.random.default_rng(42)
    all_candidates = np.arange(2**N, dtype=np.uint32).tolist()
    rng.shuffle(all_candidates)
    all_candidates = [c for c in all_candidates if c not in code_set]
    
    fast_greedy_add(code, code_set, all_candidates, deadline - 5)
    
    if time.time() < deadline - 10:
        rng2 = np.random.default_rng(137)
        order2 = np.arange(2**N, dtype=np.uint32).tolist()
        rng2.shuffle(order2)
        order2 = [c for c in order2 if c not in code_set]
        fast_greedy_add(code, code_set, order2, deadline - 5)
    
    return code
