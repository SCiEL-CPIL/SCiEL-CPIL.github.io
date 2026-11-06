import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 17
    D = 6
    
    # 1. Construct the 256-word linear code (shortened [18,9,6] QR code)
    residues = [1, 4, 5, 6, 7, 9, 11, 16, 17]
    n_qr = 18
    k_qr = 9
    
    g0 = 0
    for r in residues:
        g0 |= 1 << (r - 1)
        
    basis = []
    current = g0
    for i in range(k_qr):
        basis.append(current)
        current = (current >> 1) | ((current & 1) << (n_qr - 1))
        
    codewords_18 = []
    for mask in range(1 << k_qr):
        word = 0
        for i in range(k_qr):
            if mask & (1 << i):
                word ^= basis[i]
        codewords_18.append(word)
        
    initial_words = []
    for w in codewords_18:
        if (w & (1 << 17)) == 0:
            new_w = w & ((1 << 17) - 1)
            initial_words.append(new_w)
            
    initial_words = list(set(initial_words))
    
    max_size = 1 << N
    coverage = np.zeros(max_size, dtype=np.int32)
    
    ball_offsets = []
    for u in range(max_size):
        if u.bit_count() <= 5:
            ball_offsets.append(u)
    ball_offsets = np.array(ball_offsets, dtype=np.int32)
    
    for w in initial_words:
        neighbors = w ^ ball_offsets
        coverage[neighbors] += 1
        
    current_code = initial_words[:]
    
    def time_up():
        return time.time() > deadline
        
    while not time_up():
        candidates = np.where(coverage == 0)[0]
        if len(candidates) == 0:
            break
            
        idx = np.random.randint(len(candidates))
        new_word = int(candidates[idx])
        
        current_code.append(new_word)
        
        neighbors = new_word ^ ball_offsets
        coverage[neighbors] += 1
        
    return current_code
