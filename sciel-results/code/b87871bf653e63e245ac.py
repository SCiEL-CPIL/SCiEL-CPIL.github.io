    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        w = random.randint(0, (1 << N) - 1)
        wt = w.bit_count()
        if wt < 6 or wt > 12: continue
        # Loop over 512 codewords
