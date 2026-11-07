    lpf = list(range(max_val + 1))
    for i in range(2, isqrt(max_val) + 1):
        if lpf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
