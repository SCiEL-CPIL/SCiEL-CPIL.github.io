    for a in range(1, N + 1):
        a2 = a * a
        divisors = []
        for i in range(1, int(a2**0.5) + 1):
            if a2 % i == 0:
                divisors.append(i)
                if i != a2 // i:
                    divisors.append(a2 // i)
        
        seen_pairs = set()
        for d in divisors:
            d2 = a2 // d
            if d == d2:
                continue
            pair_key = (min(d, d2), max(d, d2))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            b = a + d
            c = a + d2
            
            if b <= N and c <= N:
                if b != c: # d != d2 ensures this
                    t = tuple(sorted((a, b, c)))
                    triples.append(t)
                    for v in t:
                        triples_of[v].append(len(triples) - 1)
