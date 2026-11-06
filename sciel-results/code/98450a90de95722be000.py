import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128

    # Precompute bad triples and conflict structure using bitsets for speed
    # conflicts[x] is a list of (u, v) pairs that cannot both be in S with x
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd(a, b)
                g_bc = gcd(b, c)
                g_ac = gcd(a, c)
                if g_ab == g_bc == g_ac:
                    bad_triples.append((a, b, c))

    conflicts = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # Also precompute for each x, the set of elements that conflict with x in some way
    # For faster validity checking, we'll use a different approach

    def is_valid_add(S_set, x):
        """Check if x can be added to S_set without creating a bad triple"""
        for (u, v) in conflicts[x]:
            if u in S_set and v in S_set:
                return False
        return True

    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        S_set = set(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if gcd(a, b) == gcd(b, c) == gcd(a, c):
                        return False
        return True

    best_set = set()
    best_size = 0

    # Multiple greedy constructions with different orderings
    random.seed(42)
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(conflicts[x])),
        sorted(range(1, N + 1), key=lambda x: -len(conflicts[x])),
        sorted(range(1, N + 1), key=lambda x: x % 7),
        sorted(range(1, N + 1), key=lambda x: x % 5),
    ]
    
    for _ in range(20):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    for order in orders:
        S = set()
        S_set = set()
        for x in order:
            if is_valid_add(S_set, x):
                S.add(x)
                S_set.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    S = best_set.copy()
    current_size = len(S)

    # Local search with multiple improvement strategies
    def try_improve(S, current_size, deadline):
        """Try to improve S through various moves. Returns (new_S, new_size, improved)"""
        S_set = set(S)
        
        # Strategy 1: Add single elements
        for x in range(1, N + 1):
            if x not in S_set:
                if is_valid_add(S_set, x):
                    S.add(x)
                    S_set.add(x)
                    return S, current_size + 1, True
        
        # Strategy 2: 1-for-1 swaps (replace one element with another)
        S_list = list(S)
        for u in S_list:
            for v in range(1, N + 1):
                if v not in S_set:
                    temp_S_set = S_set - {u}
                    if is_valid_add(temp_S_set, v):
                        S.remove(u)
                        S.add(v)
                        S_set.remove(u)
                        S_set.add(v)
                        return S, current_size, True
        
        # Strategy 3: 1-for-2 exchanges (remove one, add two)
        if S:
            S_list = list(S)
            random.shuffle(S_list)
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            
            for u in S_list[:20]:  # Try first 20 elements
                temp_S_set = S_set - {u}
                
                # Try to find two elements that can be added
                for x in candidates:
                    if is_valid_add(temp_S_set, x):
                        temp_with_x = temp_S_set | {x}
                        for y in candidates:
                            if y != x and is_valid_add(temp_with_x, y):
                                S.remove(u)
                                S.add(x)
                                S.add(y)
                                S_set = set(S)
                                return S, current_size + 1, True
                if time.time() > deadline:
                    break
        
        return S, current_size, False

    # Main local search loop
    stuck_count = 0
    while time.time() < deadline:
        S, current_size, improved = try_improve(S, current_size, deadline)
        
        if improved:
            stuck_count = 0
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
        else:
            stuck_count += 1
            if stuck_count > 50:
                # Try a more aggressive random restart approach
                # Keep the best set and try to find improvements from it
                S = best_set.copy()
                current_size = len(S)
                stuck_count = 0
                
                # Try random 1-for-1 swaps to escape local minima
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    if not S:
                        break
                    u = random.choice(list(S))
                    v = random.randint(1, N)
                    if v not in S:
                        temp_S_set = set(S) - {u}
                        if is_valid_add(temp_S_set, v):
                            S.remove(u)
                            S.add(v)
                            if len(S) > best_size:
                                best_size = len(S)
                                best_set = S.copy()

    # Final polish: try to add any remaining elements
    S_set = set(S)
    for x in range(1, N + 1):
        if x not in S_set:
            if is_valid_add(S_set, x):
                S.add(x)
                S_set.add(x)

    if len(S) > best_size:
        best_size = len(S)
        best_set = S.copy()

    return list(best_set)
