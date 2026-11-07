import time

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    N = 256
    
    # Prime factorization for all numbers 2..256
    prime_factors = [{} for _ in range(N+1)]
    for i in range(2, N+1):
        m = i
        for p in range(2, int(m**0.5)+1):
            if m % p == 0:
                cnt = 0
                while m % p == 0:
                    cnt += 1
                    m //= p
                prime_factors[i][p] = cnt
        if m > 1:
            prime_factors[i][m] = prime_factors[i].get(m, 0) + 1
    
    def divides_product(a, b, c):
        for p, exp_a in prime_factors[a].items():
            if prime_factors[b].get(p, 0) + prime_factors[c].get(p, 0) < exp_a:
                return False
        return True
    
    # Precompute conflict bitsets
    # conflict1[a][b] = bitset of c such that a|b*c
    # conflict2[a][b] = bitset of c such that b|a*c  (a is the new element)
    # conflict3[a][b] = bitset of c such that b|c*a  (a is the new element)
    
    conflict1 = [[0]*(N+1) for _ in range(N+1)]
    conflict2 = [[0]*(N+1) for _ in range(N+1)]
    conflict3 = [[0]*(N+1) for _ in range(N+1)]
    
    for a in range(2, N+1):
        for b in range(2, N+1):
            if a == b:
                continue
            for c in range(2, N+1):
                if c == a or c == b:
                    continue
                if divides_product(a, b, c):
                    conflict1[a][b] |= (1 << c)
                if divides_product(b, a, c):
                    conflict2[a][b] |= (1 << c)
                if divides_product(b, c, a):
                    conflict3[a][b] |= (1 << c)
    
    def can_add(S_bits, x):
        # Check if x can be added to S
        # Case 1: x|b*c for b,c in S, b!=c
        for b in range(2, N+1):
            if S_bits & (1 << b):
                if conflict1[x][b] & S_bits & ~(1 << b):
                    return False
        # Case 2: b|x*c for b,c in S, b!=c, b!=x, c!=x
        for b in range(2, N+1):
            if (S_bits & (1 << b)) and b != x:
                if conflict2[x][b] & S_bits & ~(1 << b) & ~(1 << x):
                    return False
        # Case 3: b|c*x for b,c in S, b!=c, b!=x, c!=x
        for b in range(2, N+1):
            if (S_bits & (1 << b)) and b != x:
                if conflict3[x][b] & S_bits & ~(1 << b) & ~(1 << x):
                    return False
        return True
    
    # Branch and bound with multiple orderings
    best_size = 0
    best_bits = 0
    
    def solve(order):
        nonlocal best_size, best_bits
        
        def bb(idx, S_bits, size):
            nonlocal best_size, best_bits
            if time.time() > deadline:
                return
            if size > best_size:
                best_size = size
                best_bits = S_bits
            if idx >= len(order):
                return
            if size + len(order) - idx <= best_size:
                return
            x = order[idx]
            # Try include
            if can_add(S_bits, x):
                bb(idx+1, S_bits | (1 << x), size+1)
            # Try exclude
            bb(idx+1, S_bits, size)
        
        bb(0, 0, 0)
    
    # Try different orderings
    orders = [
        list(range(2, N+1)),
        list(range(N, 1, -1)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        solve(order)
    
    result = [i for i in range(2, N+1) if best_bits & (1 << i)]
    
    # Verify
    for i in range(len(result)):
        for j in range(len(result)):
            if i == j:
                continue
            for k in range(len(result)):
                if i == k or j == k:
                    continue
                if divides_product(result[i], result[j], result[k]):
                    # Invalid, fallback
                    result = [i for i in range(129, 257) if all(i%p!=0 for p in range(2, int(i**0.5)+1))]
                    break
            else:
                continue
            break
        else:
            continue
        break
    
    return [int(x) for x in result]
