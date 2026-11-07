def construct():
    max_val = max(hi for _, hi in P['gaps'])
    
    # Sieve for Least Prime Factor (LPF)
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            lpf[i] = i
            if i * i <= max_val:
                for j in range(i * i, max_val + 1, i):
                    if lpf[j] == 0:
                        lpf[j] = i
    
    result = []
    for lo, hi in P['gaps']:
        gap_size = hi - lo
        witness = None
        for m in range(lo + 1, hi):
            if lpf[m] >= gap_size:
                witness = m
                break
        result.append(witness)
        
    return result
