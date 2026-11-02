<<<<<<< SEARCH
    if r < 0.30:
        L = rng.randint(4, 9)        # 4..8
    elif r < 0.60:
        L = rng.randint(9, 17)       # 9..16
    elif r < 0.85:
        L = rng.randint(17, 33)      # 17..32
    else:
        L = rng.randint(33, 65)      # 33..64
    L = min(L, N)
    start = int(rng.randint(0, N - L + 1))
=======
    if r < 0.30:
        L = int(rng.integers(4, 9))        # 4..8
    elif r < 0.60:
        L = int(rng.integers(9, 17))       # 9..16
    elif r < 0.85:
        L = int(rng.integers(17, 33))      # 17..32
    else:
        L = int(rng.integers(33, 65))      # 33..64
    L = min(L, N)
    start = int(rng.integers(0, N - L + 1))
>>>>>>> REPLACE
