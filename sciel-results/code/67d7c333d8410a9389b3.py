import numpy as np

def construct():
    # Limit for powerful numbers
    # We use M = 10^8.
    # K ~ 2.6 * sqrt(M) ~ 26000.
    # Total pairs ~ K^2/2 ~ 3.4 * 10^8.
    # We process in chunks to avoid O(K^2) memory for sums.
    M = 10**8
    
    # Generate powerful numbers up to M
    powerful_set = set()
    max_v = int(M ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        v3 = v ** 3
        max_u = int((M // v3) ** 0.5) + 1
        for u in range(1, max_u + 1):
            p = (u * u) * v3
            if p <= M:
                powerful_set.add(p)
            else:
                break
                
    powerful_list = sorted(powerful_set)
    P = np.array(powerful_list, dtype=np.int32)
    
    if len(P) < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    K = len(P)
    # Use int32 for counts to save memory. Max count is K/2 ~ 13000.
    counts = np.zeros(2 * M + 1, dtype=np.int32)
    
    # Process row by row to avoid creating a K x K array
    for i in range(K):
        # Sums of P[i] with P[i:]
        # We only need a <= b, so j >= i
        row_sums = P[i] + P[i:]
        # bincount returns int64, cast to int32 to match counts
        counts += np.bincount(row_sums, minlength=2 * M + 1).astype(np.int32)
        
    # Find N with max representations
    # N must be >= 2
    if len(counts) > 1:
        counts[0] = 0
        counts[1] = 0
        
    max_idx = np.argmax(counts)
    N = int(max_idx)
    
    # Extract pairs
    pairs = []
    P_set = set(powerful_list)
    for a in powerful_list:
        b = N - a
        if b >= a and b in P_set:
            pairs.append([a, b])
            
    return {'N': N, 'pairs': pairs}
