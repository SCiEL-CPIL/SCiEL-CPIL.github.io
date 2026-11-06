import time, random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    deadline = time.time() + 50.0 # Safety margin
    N = 128
    nums = list(range(1, N+1))
    
    # Precompute forbidden sets
    # forbidden[a][b] is a set of c such that (a,b,c) is a bad triple
    # We only need to store for a < b, and use symmetry
    forbidden = {}
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = _lcm(a, b)
            bads = set()
            for c in range(1, N+1):
                if c == a or c == b:
                    continue
                if _lcm(a, c) == lab and _lcm(b, c) == lab:
                    bads.add(c)
            forbidden[(a,b)] = bads
            
    def try_greedy(order):
        S = set()
        # To speed up conflict checking, we can maintain a list of elements in S
        # and for each new candidate x, check against all pairs in S
        # Optimization: If |S| is large, this is slow.
        # Alternative: For each x, check if x is in forbidden[(a,b)] for any a,b in S.
        # This is still O(|S|^2).
        # Since N=128, |S| max 128. 128^2 = 16384. 16384 * 128 = 2M ops per run.
        # 500 runs = 1B ops? No, the check is `if x in forbidden[...]`.
        # Set lookup is O(1).
        # So it is 2M set lookups per greedy run.
        # 500 runs = 1B lookups. Python might be slow.
        
        # Let's optimize the inner loop.
        slist = []
        for x in order:
            ok = True
            # Check x against all pairs in current S
            for i in range(len(slist)):
                a = slist[i]
                for j in range(i+1, len(slist)):
                    b = slist[j]
                    key = (a,b) if a < b else (b,a)
                    if x in forbidden[key]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                S.add(x)
                slist.append(x)
        return S

    best = set()
    
    # Try a few deterministic orders first
    # 1. Natural order
    S = try_greedy(nums)
    if len(S) > len(best): best = set(S)
    
    # 2. Reverse order
    S = try_greedy(nums[::-1])
    if len(S) > len(best): best = set(S)
    
    # 3. Order by number of divisors (more composite first? or prime first?)
    # Primes are "independent" in many ways. Let's try primes first.
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
        
    primes = [x for x in nums if is_prime(x)]
    composites = [x for x in nums if not is_prime(x)]
    S = try_greedy(primes + composites)
    if len(S) > len(best): best = set(S)
    
    # Randomized restarts
    random.seed(42)
    start_time = time.time()
    restarts = 0
    while time.time() < deadline:
        order = nums[:]
        random.shuffle(order)
        S = try_greedy(order)
        if len(S) > len(best):
            best = set(S)
        restarts += 1
        
    if not best:
        best = {1}
    return sorted(best)
