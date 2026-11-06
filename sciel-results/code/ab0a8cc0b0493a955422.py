import time
import random
from math import gcd

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    N = 256
    # Sieve primes up to N
    is_prime = [True]*(N+1)
    is_prime[0]=is_prime[1]=False
    for i in range(2,int(N**0.5)+1):
        if is_prime[i]:
            for j in range(i*i,N+1,i):
                is_prime[j]=False
    primes = [i for i in range(2,N+1) if is_prime[i]]

    # Precompute pairwise coprime check for all pairs up to N
    # cop[i][j] = 1 if gcd(i,j)==1 else 0
    cop = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        cop[i][i] = 0
        for j in range(i+1, N+1):
            cop[i][j] = 1 if gcd(i,j)==1 else 0
            cop[j][i] = cop[i][j]

    def evaluate(A,B):
        """Return number of distinct sums if pairwise coprime, else 0."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a+b)
        lst = sorted(sums)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if cop[lst[i]][lst[j]] == 0:
                    return 0
        return len(lst)

    # Seed
    best = ([1,7,13],[4,10,16])
    best_card = evaluate(best[0],best[1])
    if best_card <= 0:
        best_card = 5

    rng = random.Random(20240517)

    # === Phase 1: Structured search for cardinality 9 ===
    # Try to find A,B such that all 9 sums are distinct and pairwise coprime.
    # Strategy: enumerate possible sum sets (9 primes), then solve for A,B.
    
    # For 9 distinct sums from 3x3 grid: sums = {a_i + b_j}
    # If we fix a_1=1 (WLOG shift), then b_j = s_{1j} - 1, a_i = s_{i1} - b_1
    # Actually let's use the offset approach: pick a base prime P, 
    # offsets r2,r3 for A and c2,c3 for B such that all P+r+c are prime.
    
    found9 = False
    
    # Method: iterate over candidate "anchor" primes and offset patterns
    for P in primes:
        if time.time() > deadline:
            break
        # Offsets for A: 0, r2, r3 (so A = [x, x+r2, x+r3])
        # Offsets for B: 0, c2, c3 (so B = [P-x, P-x+c2, P-x+c3])
        # Sum grid: P + r + c for r in {0,r2,r3}, c in {0,c2,c3}
        # Need all 9 to be prime (or at least pairwise coprime)
        
        # Collect valid offsets
        S = [x for x in range(-127,128) if x != 0 and 2 <= P+x <= N and is_prime[P+x]]
        if len(S) < 4:
            continue
        
        # Try pairs of offsets for A
        for idx2 in range(len(S)):
            if found9: break
            if time.time() > deadline: break
            r2 = S[idx2]
            for idx3 in range(idx2+1, len(S)):
                if found9: break
                if time.time() > deadline: break
                r3 = S[idx3]
                # Check P+r2 and P+r3 are prime (already ensured by S)
                # Now find c2, c3 such that P+r2+c2, P+r2+c3, P+r3+c2, P+r3+c3 are all prime
                valid_c = []
                for c in S:
                    if c == r2 or c == r3:
                        continue
                    v1 = P + r2 + c
                    v2 = P + r3 + c
                    if 2 <= v1 <= N and 2 <= v2 <= N and is_prime[v1] and is_prime[v2]:
                        valid_c.append(c)
                
                if len(valid_c) < 2:
                    continue
                
                for ci in range(len(valid_c)):
                    if found9: break
                    c2 = valid_c[ci]
                    for cj in range(ci+1, len(valid_c)):
                        if found9: break
                        c3 = valid_c[cj]
                        
                        nine = [P, P+c2, P+c3, P+r2, P+r3, 
                                P+r2+c2, P+r2+c3, P+r3+c2, P+r3+c3]
                        if len(set(nine)) != 9:
                            continue
                        
                        # All 9 are prime by construction, so pairwise coprime
                        # Now find x such that A=[x, x+r2, x+r3] and B=[P-x, P-x+c2, P-x+c3]
                        # are all in [1,128] and distinct
                        r_list = [0, r2, r3]
                        c_list = [0, c2, c3]
                        rmin = min(r_list); rmax = max(r_list)
                        cmin = min(c_list); cmax = max(c_list)
                        
                        # A: x in [1,128], x+r2 in [1,128], x+r3 in [1,128]
                        # => x in [1-rmin, 128-rmax]
                        # B: P-x in [1,128], P-x+c2 in [1,128], P-x+c3 in [1,128]
                        # => P-x in [1-cmin, 128-cmax] => x in [P-128+cmax, P-1+cmin]
                        
                        lo = max(1, 1-rmin, P-128+cmax)
                        hi = min(128, 128-rmax, P-1+cmin)
                        
                        if lo <= hi:
                            # Try a few values of x
                            x_candidates = list(range(lo, hi+1))
                            rng.shuffle(x_candidates)
                            for x in x_candidates[:20]:
                                A = [x, x+r2, x+r3]
                                B = [P-x, P-x+c2, P-x+c3]
                                if all(1<=v<=128 for v in A) and all(1<=v<=128 for v in B):
                                    if len(set(A))==3 and len(set(B))==3:
                                        card = evaluate(A,B)
                                        if card == 9:
                                            best = (A[:], B[:])
                                            best_card = 9
                                            found9 = True
                                            break
                            if found9:
                                break
                    if found9:
                        break
                if found9:
                    break
            if found9:
                break
        if found9:
            break

    if best_card >= 9:
        return best

    # === Phase 2: Broader search for cardinality 9 with non-prime coprime sums ===
    # Some pairwise coprime sets include composites like 4, 9, 25, 49, 121, etc.
    # Build list of "coprime-compatible" values: primes and prime powers
    prime_powers = set(primes)
    for p in primes:
        pp = p*p
        while pp <= N:
            prime_powers.add(pp)
            pp *= p
    prime_powers = sorted(prime_powers)
    
    # Check pairwise coprimality among prime_powers
    # Two prime powers are coprime iff they are powers of different primes
    pp_prime = {}  # value -> prime base
    for v in prime_powers:
        for p in primes:
            if v % p == 0:
                pp_prime[v] = p
                break
    
    def pp_coprime(a, b):
        if a == b:
            return False
        return pp_prime.get(a) != pp_prime.get(b)

    # Try to find 9 values from prime_powers that form a valid 3x3 sum grid
    # This is more complex; use randomized search guided by structure
    
    # === Phase 3: Randomized local search ===
    def rand_set():
        while True:
            s = [rng.randint(1,128) for _ in range(3)]
            if len(set(s)) == 3:
                return s

    it = 0
    while time.time() < deadline:
        it += 1
        A = rand_set()
        B = rand_set()
        card = evaluate(A,B)
        if card > best_card:
            best = (A[:], B[:])
            best_card = card
            if best_card >= 9:
                break
        
        # Local search: try mutations
        if card > 0:
            for _ in range(30):
                if time.time() > deadline:
                    break
                A2 = A[:]
                B2 = B[:]
                if rng.random() < 0.5:
                    A2[rng.randint(0,2)] = rng.randint(1,128)
                else:
                    B2[rng.randint(0,2)] = rng.randint(1,128)
                if len(set(A2)) < 3 or len(set(B2)) < 3:
                    continue
                c2 = evaluate(A2,B2)
                if c2 > best_card:
                    best = (A2[:], B2[:])
                    best_card = c2
                    A, B = A2, B2
                    if best_card >= 9:
                        break

    return best
