for a in range(n * m): # i1, j1
    for b in range(m * p): # j2, k1
        if j1 != j2: continue
        for c in range(p * n): # k2, i2
            if i1 == i2 and k1 == k2: continue
            actual = ...
            if actual != 0: return False
