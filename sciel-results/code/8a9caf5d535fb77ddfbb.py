def construct():
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def max_exponents_for_prime(p, limit=40):
        max_exp = 0
        val = p
        while val <= limit:
            max_exp += 1
            val *= p
        # Find max subset of {1..max_exp} with distinct subset sums
        # Brute force is fine for small max_exp
        best_size = 0
        best_set = []
        # Iterate through all subsets of {1..max_exp}
        # Actually, we can just check sizes from max_exp down to 1
        # But max_exp is small.
        import itertools
        for r in range(max_exp, 0, -1):
            found = False
            for combo in itertools.combinations(range(1, max_exp + 1), r):
                # Check if subset sums are distinct
                sums = set()
                valid = True
                # Generate all subset sums of combo
                # 2^r sums
                for mask in range(1 << r):
                    s = 0
                    for i in range(r):
                        if mask & (1 << i):
                            s += combo[i]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if valid:
                    best_set = list(combo)
                    found = True
                    break
            if found:
                return best_set
        return []

    result = []
    for p in primes:
        exps = max_exponents_for_prime(p)
        for e in exps:
            result.append(p ** e)
            
    return result
