    def construct():
        import time
        deadline = time.time() + 85
        max_val = max(hi for lo, hi in P['gaps'])
        lpf = list(range(max_val + 1))
        for i in range(2, int(max_val**0.5) + 1):
            if lpf[i] == i:
                for j in range(i*i, max_val + 1, i):
                    if lpf[j] == j:
                        lpf[j] = i
        result = [None] * len(P['gaps'])
        for i, (lo, hi) in enumerate(P['gaps']):
            gap = hi - lo
            for m in range(lo + 1, hi):
                if lpf[m] >= gap:
                    result[i] = m
                    break
            if time.time() > deadline:
                break
        # ...
        return final_result
    