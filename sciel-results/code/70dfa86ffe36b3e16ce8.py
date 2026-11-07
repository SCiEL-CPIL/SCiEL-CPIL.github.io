        # Initialize
        # Pick a random window
        W_init = 50
        A = random.randint(1, MAX_VAL - W_init)
        vals = [0] * n
        valid_init = False
        attempts = 0
        while not valid_init and attempts < 100:
            attempts += 1
            seen = set()
            for i in range(n):
                d = divisors[i]
                # Find multiples in [A, A+W_init]
                k_start = (A + d - 1) // d
                k_end = (A + W_init) // d
                if k_start > k_end:
                    valid_init = False
                    break
                k = random.randint(k_start, k_end)
                v = k * d
                if v in seen:
                    valid_init = False
                    break
                seen.add(v)
                vals[i] = v
            if valid_init:
                break
        if not valid_init:
            continue # Skip this restart
