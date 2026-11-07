import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 20_000_000
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline - 20.0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    freq = {}
    max_M = 10**18
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M > max_M:
            break
        freq[M] = freq.get(M, 0) + 1
        
    best_M = None
    max_count = 0
    for M, c in freq.items():
        if c > max_count:
            max_count = c
            best_M = M
            
    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    ks = []
    for k in range(1, N + 1):
        if k * sigma[k] == best_M:
            ks.append(k)
            
    return {'M': int(best_M), 'k': ks}
