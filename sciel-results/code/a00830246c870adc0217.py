import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Quadratic residues mod 19
    Q = {1, 4, 5, 6, 7, 9, 11, 16, 17}
    v = np.zeros(19, dtype=np.int8)
    v[0] = 1
    for q in Q:
        v[q] = 1
        
    # Generate circulant matrix
    G_full = np.zeros((19, 19), dtype=np.int8)
    for i in range(19):
        G_full[i] = np.roll(v, i)
        
    # Find basis for the row space (rank 10)
    basis_set = []
    for i in range(19):
        r = tuple(G_full[i])
        if len(basis_set) >= 10:
            break
        # Check if r is independent of basis_set
        M = np.array(basis_set + [r], dtype=np.int8)
        # Compute rank over GF(2)
        rank = 0
        M_copy = M.copy()
        for col in range(19):
            pivot = -1
            for row in range(rank, len(M_copy)):
                if M_copy[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            M_copy[rank], M_copy[pivot] = M_copy[pivot].copy(), M_copy[rank].copy()
            for row in range(len(M_copy)):
                if row != rank and M_copy[row, col] == 1:
                    M_copy[row] ^= M_copy[rank]
            rank += 1
        if rank == len(basis_set) + 1:
            basis_set.append(r)
            
    if len(basis_set) < 10:
        # Fallback to seed
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    G = np.array(basis_set, dtype=np.int8)
    
    # Generate all 2^10 = 1024 codewords of the [19,10,7] code
    codewords_19 = []
    for mask in range(1024):
        cw = np.zeros(19, dtype=np.int8)
        for i in range(10):
            if (mask >> i) & 1:
                cw ^= G[i]
        codewords_19.append(cw)
        
    # Shorten to 18 bits: fix coord 0 to 0
    codewords_18 = [cw[1:] for cw in codewords_19 if cw[0] == 0]
    
    # Shorten to 17 bits: fix coord 0 of the 18-bit code to 0
    codewords_17 = [cw[1:] for cw in codewords_18 if cw[0] == 0]
    
    words = []
    for cw in codewords_17:
        w = 0
        for i in range(17):
            if cw[i] == 1:
                w |= (1 << i)
        words.append(w)
        
    words = list(set(words))
    
    # Greedy expansion
    if time.time() < deadline - 10:
        word_set = set(words)
        current_words = words[:]
        
        # Multiple passes to add more words
        added_any = True
        while added_any and time.time() < deadline - 5:
            added_any = False
            for cand in range(2**17):
                if cand in word_set:
                    continue
                valid = True
                for w in current_words:
                    if (cand ^ w).bit_count() < 6:
                        valid = False
                        break
                if valid:
                    current_words.append(cand)
                    word_set.add(cand)
                    added_any = True
                    
    return current_words
