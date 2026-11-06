import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 17
    TotalWords = 1 << N
    
    # 1. Construct [17, 8, 6] QR Code
    # Generator polynomial for [18, 9, 6] QR code
    g_coeffs = [1, 1, 1, 0, 0, 1, 1, 1, 0, 1]
    
    qr_words = []
    for msg in range(512):
        c = 0
        for i in range(9):
            if (msg >> i) & 1:
                g_shifted = 0
                for j in range(10):
                    if g_coeffs[j]:
                        g_shifted |= (1 << (i + j))
                c ^= g_shifted
        qr_words.append(c)
        
    linear_words = []
    for c in qr_words:
        if (c >> 17) & 1 == 0:
            linear_words.append(c & ((1 << 17) - 1))
            
    linear_words = list(set(linear_words))
    
    # 2. Precompute AllWords
    AllWords = np.zeros((TotalWords, N), dtype=np.uint8)
    i = np.arange(TotalWords)
    for j in range(N):
        AllWords[:, j] = (i >> j) & 1
        
    weights = np.sum(AllWords, axis=1)
    
    # 3. Greedy Addition with multiple orders
    best_set = set(linear_words)
    
    def greedy_add(order):
        S = set(linear_words)
        S_list = list(S)
        S_np = AllWords[S_list].copy()
        
        for idx in order:
            if time.time() > deadline:
                break
            if idx in S:
                continue
            w_bits = AllWords[idx]
            dists = np.sum(S_np != w_bits, axis=1)
            if np.min(dists) >= 6:
                S.add(idx)
                S_np = np.vstack([S_np, w_bits])
                
        return S

    # Order 1: Natural
    S1 = greedy_add(range(TotalWords))
    if len(S1) > len(best_set):
        best_set = S1
        
    # Order 2: Weight Ascending
    if time.time() < deadline:
        order_w_asc = np.argsort(weights)
        S2 = greedy_add(order_w_asc)
        if len(S2) > len(best_set):
            best_set = S2
            
    # Order 3: Weight Descending
    if time.time() < deadline:
        order_w_desc = np.argsort(-weights)
        S3 = greedy_add(order_w_desc)
        if len(S3) > len(best_set):
            best_set = S3
            
    # Order 4: Random Shuffles
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        order_rand = np.random.permutation(TotalWords)
        S_rand = greedy_add(order_rand)
        if len(S_rand) > len(best_set):
            best_set = S_rand
            
    return list(best_set)
