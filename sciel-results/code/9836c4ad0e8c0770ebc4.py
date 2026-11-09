def construct():
    n = 180
    S = set()
    
    # Precompute for each k, all pairs (a,b) with a<b, a,b in [1,n], a!=b
    # such that (a-k)(b-k) = k^2, i.e., ab/(a+b) = k
    conflict_pairs = {}
    for k in range(1, n + 1):
        k2 = k * k
        pairs = set()
        for d in range(1, k2 + 1):
            if k2 % d == 0:
                e = k2 // d
                a = k + d
                b = k + e
                if a == b:
                    continue
                if 1 <= a <= n and 1 <= b <= n:
                    pairs.add((min(a, b), max(a, b)))
        conflict_pairs[k] = pairs
    
    # Greedy: add numbers from n down to 1 (larger numbers are safer)
    for k in range(n, 0, -1):
        conflict = False
        for (a, b) in conflict_pairs[k]:
            if a in S and b in S:
                conflict = True
                break
        if not conflict:
            S.add(k)
    
    # Verify correctness
    S_list = sorted(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a, b = S_list[i], S_list[j]
            num = a * b
            den = a + b
            if num % den == 0:
                c = num // den
                if c in S and c != a and c != b:
                    # Fallback to seed solution
                    return list(range(91, 181))
    
    return S_list
