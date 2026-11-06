import time
from math import gcd

N = 128
DEADLINE_MARGIN = 5.0  # seconds of safety margin

# Precompute LCM table
lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, N + 1):
        lcm_table[i][j] = i * j // gcd(i, j)

# For each element x, precompute the set of pairs (a, b) such that
# {a, b, x} is a bad triple, i.e., lcm(a,b) == lcm(a,x) == lcm(b,x)
# We'll store this as a list of frozensets for fast lookup.
# Actually, let's think differently.
# For a given set S, adding x is invalid if there exist a, b in S such that
# lcm(a,b) == lcm(a,x) == lcm(b,x).
# Let L_x = lcm(a,x) = lcm(b,x) = lcm(a,b).
# This means lcm(a,x) = lcm(b,x). Let's call this common value L.
# Then lcm(a,b) must also equal L.
# Note that lcm(a,x) = L implies L is a multiple of x and a.
# Actually, a simpler approach: for each x, precompute all pairs (a,b) that conflict with x.
# conflicts_with_x[x] = set of (a, b) pairs such that adding x to {a,b} creates a bad triple.

conflicts_with_x = [set() for _ in range(N + 1)]
for x in range(1, N + 1):
    for a in range(1, N + 1):
        if a == x:
            continue
        for b in range(a + 1, N + 1):
            if b == x:
                continue
            # Check if {a, b, x} is a bad triple
            L_ab = lcm_table[a][b]
            L_ax = lcm_table[a][x]
            L_bx = lcm_table[b][x]
            if L_ab == L_ax == L_bx:
                conflicts_with_x[x].add((a, b))

# For faster checking, let's also build for each x a set of "conflict partners"
# Actually, the check is: for x to be added to S, we need no pair (a,b) in conflicts_with_x[x]
# where both a and b are in S.
# We can check this by iterating over conflicts_with_x[x] and checking if both elements are in S.

def can_add(x, S):
    """Check if x can be added to set S without creating a bad triple."""
    for a, b in conflicts_with_x[x]:
        if a in S and b in S:
            return False
    return True

def greedy(order, deadline):
    """Greedy construction using a specific order."""
    S = set()
    for x in order:
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)
    return S

def local_search(S, deadline):
    """Local search to improve the set size."""
    S = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any element
        for x in range(1, N + 1):
            if x not in S:
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    break
        if not improved:
            # Try removing one element and adding others
            # This is a 1-swap: remove y, then greedily add
            # But this doesn't directly increase size unless we can add more than 1
            # Let's try: for each y in S, remove it, then greedily add as many as possible
            # If the resulting set is larger, keep it.
            best_trial = S
            best_size = len(S)
            for y in list(S):
                if time.time() > deadline:
                    break
                trial = S - {y}
                # Greedily add elements
                for x in range(1, N + 1):
                    if x not in trial:
                        if can_add(x, trial):
                            trial.add(x)
                if len(trial) > best_size:
                    best_size = len(trail)
                    best_trial = trial
            if len(best_trial) > len(S):
                S = best_trial
                improved = True
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN  # 85 seconds
    
    import random
    random.seed(12345)
    
    best_set = set()
    best_size = 0
    
    # Generate multiple orders
    base = list(range(1, N + 1))
    
    # Order 1: natural order
    orders = [base]
    
    # Order 2: reverse natural
    orders.append(base[::-1])
    
    # Order 3: by number of prime factors (ascending) - "simpler" numbers first
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    orders.append(sorted(base, key=num_prime_factors))
    orders.append(sorted(base, key=num_prime_factors, reverse=True))
    
    # Order 4: by number of divisors
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    orders.append(sorted(base, key=num_divisors))
    orders.append(sorted(base, key=num_divisors, reverse=True))
    
    # Order 5-14: random shuffles
    for _ in range(10):
        shuffled = base.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    # Run greedy for each order
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search on the best set
    if time.time() < deadline:
        improved_set = local_search(best_set, deadline)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(improved_set)
    
    # Additional random restarts with local search if time permits
    restart_count = 0
    while time.time() < deadline and restart_count < 50:
        restart_count += 1
        shuffled = base.copy()
        random.shuffle(shuffled)
        S = greedy(shuffled, deadline)
        if len(S) > best_size:
            S = local_search(S, deadline)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    return list(best_set)

# Ensure non-empty
result = construct()
if not result:
    construct = lambda: [1]
