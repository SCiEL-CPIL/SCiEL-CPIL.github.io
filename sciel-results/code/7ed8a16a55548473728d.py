def construct():
    N = 123457
    # Sieve for largest prime factor
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i] = i
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_a = 1
    best_b = N - 1
    best_score = max(lpf[best_a], lpf[best_b])
    
    for a in range(1, N):
        b = N - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
