    last_improvement = time.time()
    while time.time() < deadline:
        # ... moves ...
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            last_improvement = time.time()
            
        if time.time() - last_improvement > 5.0:
            # Restart
            # Maybe try a different greedy order?
            # Or just reset S to greedy?
            # Let's just reset to greedy to try a different path.
            S = []
            InS = [False] * (N + 1)
            PairsInS = set()
            # Randomize greedy order?
            # The greedy is deterministic 1..128.
            # Let's shuffle the range.
            order = list(range(1, N + 1))
            random.shuffle(order)
            for x in order:
                if not (PairsInS & BadPairs[x]):
                    S.append(x)
                    InS[x] = True
                    for a in S[:-1]:
                        add_pair(a, x, PairsInS)
            last_improvement = time.time()
