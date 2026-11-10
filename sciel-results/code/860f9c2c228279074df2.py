def construct():
    MOD = 101
    # Modular inverses of 1..10 modulo 101
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, MOD - 2, MOD)

    # DP over subsets: subset_sum[s] = sum of inverses mod 101 for elements in s
    # We iterate over all 2^10 = 1024 subsets
    # For each subset, compute its sum and record the minimum size for each residue

    best_len = [11] * MOD
    best_mask = [0] * MOD

    # Iterate over all non-empty subsets
    for mask in range(1, 1 << 10):
        s = 0
        popcount = 0
        for i in range(10):
            if mask & (1 << i):
                s = (s + invs[i + 1]) % MOD
                popcount += 1

        if popcount < best_len[s]:
            best_len[s] = popcount
            best_mask[s] = mask

    # Convert masks to lists of denominators
    result = [None] * MOD
    for r in range(MOD):
        m = best_mask[r]
        lst = []
        for i in range(10):
            if m & (1 << i):
                lst.append(i + 1)
        result[r] = lst

    return result
