import numpy as np

def construct():
    M = 10**9
    
    # Generate squarefree numbers up to M^(1/3)
    max_b = int(M ** (1/3)) + 2
    
    is_sqfree = np.ones(max_b + 1, dtype=bool)
    is_sqfree[0] = False
    for p in range(2, int(max_b**0.5) + 1):
        p2 = p * p
        for j in range(p2, max_b + 1, p2):
            is_sqfree[j] = False
    
    sqfree = [b for b in range(1, max_b + 1) if is_sqfree[b]]
    
    # Generate powerful numbers: n = b^3 * c^2 <= M, b squarefree
    powerful = set()
    for b in sqfree:
        b3 = b ** 3
        if b3 > M:
            break
        max_c = int((M // b3) ** 0.5) + 1
        for c in range(1, max_c + 1):
            n = b3 * c * c
            if n <= M:
                powerful.add(n)
            else:
                break
    
    powerful_list = sorted(powerful)
    K = len(powerful_list)
    P = np.array(powerful_list, dtype=np.int64) # Use int64 to be safe for sums? No, int32 is enough for N up to 2e9? 2e9 < 2^31.
    # Actually, 2 * 10^9 is 2000000000. 2^31-1 is 2147483647. It fits.
    # But let's use int64 for P to avoid overflow in sums if we change M?
    # If M=10^9, max sum is 2*10^9. Fits in int32.
    # Let's stick to int32 for P to save memory? 70k * 4 bytes = 280KB. Negligible.
    # Let's use int64 for P to be safe.
    P = np.array(powerful_list, dtype=np.int64)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    max_N = 2 * M
    # Allocate counts array. 2*10^9 int32 is 8GB.
    # This is the risk.
    counts = np.zeros(max_N + 1, dtype=np.int32)
    
    # Chunked upper-triangular sums
    # To reduce memory peak of 'sums' and 'ms', use small chunks.
    chunk_size = 100
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        chunk = P[start:end]
        cl = end - start
        # sums is C x K
        sums = chunk[:, None] + P[None, :]
        lr = np.arange(cl)
        gr = start + lr
        cols = np.arange(K)
        mask = cols[None, :] >= gr[:, None]
        ms = sums[mask]
        
        # bincount
        # If ms is empty, skip
        if len(ms) > 0:
            counts += np.bincount(ms, minlength=max_N + 1).astype(np.int32)
    
    counts[:2] = 0
    max_idx = np.argmax(counts)
    N = int(max_idx)
    
    P_set = set(powerful_list)
    pairs = []
    for a in powerful_list:
        b = N - a
        if b >= a and b in P_set:
            pairs.append([a, b])
    
    return {'N': N, 'pairs': pairs}
