import time

def construct():
    N = 123457
    deadline = time.time() + 85  # 85s deadline to stay under 90s limit

    # Precompute largest prime factor (LPF) for all numbers up to N
    # Using a sieve-like approach: for each prime p, mark multiples with p as LPF
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for p in range(2, N + 1):
        if lpf[p] == 0:  # p is prime
            for j in range(p, N + 1, p):
                lpf[j] = p

    # O(N) scan: for each a in [1, N//2], compute max(lpf[a], lpf[N-a])
    # and track the minimum. This finds the optimal pair in one pass.
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
