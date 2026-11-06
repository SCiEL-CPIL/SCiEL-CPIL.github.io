def construct():
    from math import gcd

    N = 512

    # Precompute lcm table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            l = i * j // gcd(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # Precompute divisors for all numbers
    divisors = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            divisors[j].append(i)

    def get_divisors(n):
        return divisors[n]

    best_set = []
    best_size = 0

    # Try all M in 1..N as base
    for M in range(1, N + 1):
        base = list(divisors[M])
        # Sort base to ensure deterministic greedy
        base.sort()

        current = set(base)

        # Greedy expansion
        # Try to add elements from 1 to N not in current
        for x in range(1, N + 1):
            if x in current:
                continue

            # Check if x is compatible with all in current
            valid = True
            for y in current:
                if lcm_table[x][y] > N:
                    valid = False
                    break

            if valid:
                current.add(x)

        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)

    if not best_set:
        return [1]

    return best_set
