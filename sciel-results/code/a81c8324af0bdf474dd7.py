import time
import math

def construct():
    start_time = time.time()
    LIMIT = 90
    MARGIN = 3

    N_MAX = 1000001
    START = 10000

    # 1. Smallest Prime Factor sieve
    spf = list(range(N_MAX + 1))
    for i in range(2, int(N_MAX ** 0.5) + 1):
        if spf[i] == i:
            step = i
            for j in range(i * i, N_MAX + 1, step):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute powerful part P[k] for all k in [1, N_MAX]
    P = [1] * (N_MAX + 1)
    P[0] = 1
    P[1] = 1

    for i in range(2, N_MAX + 1):
        p = spf[i]
        x = i // p
        if x % p == 0:
            # p divides x, so p^2 divides i
            # Remove all factors of p from x to get the rest
            while x % p == 0:
                x //= p
            # Count how many p's were in i
            # We know i = p^e * x_final
            # The powerful part contribution is p^e
            # P[i] = P[x_final] * p^e
            # But we can do it recursively
            # P[i] = P[x] * p if x has p, else P[x] * p^2? No.
            # Let's use the recurrence:
            # Let i = p^k * m where m % p != 0
            # P[i] = P[m] * p^k if k >= 2 else P[m]
            
            # Actually, the recurrence used in the seed is:
            # If x % p == 0:
            #   P[i] = P[x] * p  (if x has p^1, then i has p^2, P[x] has p^1? No P[x] is powerful part of x)
            #   Wait, if x = p^j * m, j>=1.
            #   P[x] = P[m] * p^j (if j>=2) else P[m].
            #   i = p * x = p^(j+1) * m.
            #   If j=1 (x has p^1), P[x] = P[m]. i has p^2. P[i] should be P[m] * p^2.
            #   Seed logic:
            #   if x % p == 0:
            #     y = x // p
            #     if y % p == 0: # i has p^3+
            #       P[i] = P[x] * p
            #     else: # i has p^2
            #       P[i] = P[x] * p * p
            #   else:
            #     P[i] = P[x]
            
            # This is correct. Let's stick to it for speed and simplicity.
            y = x // p
            if y % p == 0:
                P[i] = P[x] * p
            else:
                P[i] = P[x] * p * p
        else:
            P[i] = P[x]

    if time.time() - start_time > LIMIT - MARGIN:
        return 10000

    # 3. Search for max P[n] * P[n+1]
    best_n = START
    best_score = -1

    # Vectorize the search?
    # scores = P[START:1000001] * P[START+1:1000001+1]
    # import numpy as np
    # scores = np.array(P[START:1000001]) * np.array(P[START+1:1000002])
    # max_idx = np.argmax(scores)
    # best_n = START + max_idx
    # best_score = scores[max_idx]

    # But we must ensure P[1000001] is computed. It is.
    # Let's try numpy for speed.
    try:
        import numpy as np
        # P is a list. Convert to array once.
        P_arr = np.array(P)
        # We need n in [10000, 1000000]
        # P[n] is P_arr[n]
        # P[n+1] is P_arr[n+1]
        # Slices:
        left = P_arr[START:1000001]
        right = P_arr[START+1:1000002]
        scores = left * right
        max_idx = np.argmax(scores)
        best_n = int(START + max_idx)
        best_score = int(scores[max_idx])
    except ImportError:
        for n in range(START, 1000001):
            score = P[n] * P[n + 1]
            if score > best_score:
                best_score = score
                best_n = n

    return int(best_n)
