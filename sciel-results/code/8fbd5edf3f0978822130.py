import time

def construct():
    N = 123457
    deadline = time.time() + 80  # 80s deadline to stay well under 90s limit
    
    # Precompute largest prime factor (LPF) for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for p in range(2, N + 1):
        if lpf[p] == 0:
            for j in range(p, N + 1, p):
                lpf[j] = p
    
    # Exhaustive O(N) scan: find pair (a, b) with a+b=N minimizing max(P(a), P(b))
    best_score = float('inf')
    best_a = 1
    best_b = N - 1
    
    mid = N // 2
    for a in range(1, mid + 1):
        if time.time() > deadline:
            break
        b = N - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
    
    return [best_a, best_b]
