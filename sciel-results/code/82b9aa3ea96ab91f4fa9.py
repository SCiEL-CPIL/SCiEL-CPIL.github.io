import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    p = 101
    denominators = list(range(1, 11))
    n = 10

    # Precompute modular inverses
    inverses = [pow(x, -1, p) for x in denominators]

    # best[r] = (min_size, mask) for residue r
    # We iterate through all subsets and keep the smallest size for each residue.
    # This effectively minimizes the maximum row length as well.
    
    best = [None] * p

    # Iterate all non-empty subsets (1023 total)
    for mask in range(1, 1 << n):
        s = 0
        sz = 0
        for i in range(n):
            if mask & (1 << i):
                s += inverses[i]
                sz += 1
        r = s % p
        if best[r] is None or sz < best[r][0]:
            best[r] = (sz, mask)

    # Verify all residues are covered
    for r in range(p):
        if best[r] is None:
            # Fallback: find any subset for missing residue
            for mask in range(1, 1 << n):
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += inverses[i]
                if s % p == r:
                    best[r] = (bin(mask).count('1'), mask)
                    break
            if best[r] is None:
                # Should never happen
                best[r] = (1, 1)

    # Build result
    result = []
    for r in range(p):
        sz, mask = best[r]
        combo = [denominators[i] for i in range(n) if mask & (1 << i)]
        result.append(combo)

    # Verify correctness
    for r in range(p):
        s = sum(inverses[d - 1] for d in result[r])
        if s % p != r:
            raise AssertionError(f"Residue {r}: got {s % p}")

    return result
