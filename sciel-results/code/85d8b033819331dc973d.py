import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128

    # Precompute bad triples and conflicts
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd(a, b)
                g_bc = gcd(b, c)
                g_ac = gcd(a, c)
                if g_ab == g_bc == g_ac:
                    bad_triples.append((a, b, c))

    # conflicts[x] = list of (u, v) such that (u, v, x) is a bad triple
    conflicts = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # Precompute conflict sets for faster membership testing
    conflict_sets = {x: set() for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for (u, v) in conflicts[x]:
            conflict_sets[x].add((u, v))
            conflict_sets[x].add((v, u))

    def is_valid_add(S_set, x):
        for (u, v) in conflicts[x]:
            if u in S_set and v in S_set:
                return False
        return True

    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if gcd(a, b) == gcd(b, c) == gcd(a, c):
                        return False
        return True

    best_set = set()
    best_size = 0

    # Greedy Construction with multiple strategies
    random.seed(42)
    
    # Compute degree (number of conflicts) for each element
    degree = {x: len(conflicts[x]) for x in range(1, N + 1)}
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    prime_flag = {x: is_prime(x) for x in range(1, N + 1)}
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: degree[x]),
        sorted(range(1, N + 1), key=lambda x: -degree[x]),
        sorted(range(1, N + 1), key=lambda x: (degree[x], -x)),
        sorted(range(1, N + 1), key=lambda x: (-degree[x], x)),
        sorted(range(1, N + 1), key=lambda x: (-prime_flag[x], degree[x])),
        sorted(range(1, N + 1), key=lambda x: (prime_flag[x], -degree[x])),
        sorted(range(1, N + 1), key=lambda x: (x % 2, degree[x])),
        sorted(range(1, N + 1), key=lambda x: (x % 3, degree[x])),
    ]
    
    # Add random orders
    for _ in range(20):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    for order in orders:
        S = set()
        for x in order:
            if is_valid_add(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    S = best_set.copy()
    current_size = len(S)

    # Local Search with multiple neighborhoods
    # N0: Add 1 element
    # N1: 1-for-2 exchange (remove 1, add 2, size +1)
    # N2: 2-for-3 exchange (remove 2, add 3, size +1)
    # N3: 1-for-1 swap (size same, to escape local minima)

    def try_add_one(S_set):
        """Try to add one element. Return (new_S, x) if successful, else (None, None)."""
        for x in range(1, N + 1):
            if x not in S_set:
                if is_valid_add(S_set, x):
                    new_S = S_set.copy()
                    new_S.add(x)
                    return new_S, x
        return None, None

    def try_swap_1_for_2(S_set, max_attempts=200):
        """Try to remove 1 element and add 2. Return (new_S, u, x, y) if successful, else (None, None, None, None)."""
        S_list = list(S_set)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S_set]
        random.shuffle(candidates)
        
        attempts = 0
        for u in S_list:
            if attempts >= max_attempts:
                break
            temp_S = S_set - {u}
            
            # Find all candidates that can be added to temp_S
            addable = []
            for x in candidates:
                if is_valid_add(temp_S, x):
                    addable.append(x)
            
            # Try pairs from addable
            for i in range(len(addable)):
                attempts += 1
                if attempts >= max_attempts:
                    break
                for j in range(i + 1, len(addable)):
                    x = addable[i]
                    y = addable[j]
                    temp_S_with_x = temp_S | {x}
                    if is_valid_add(temp_S_with_x, y):
                        new_S = temp_S | {x, y}
                        return new_S, u, x, y
        return None, None, None, None

    def try_swap_2_for_3(S_set, max_attempts=100):
        """Try to remove 2 elements and add 3. Return (new_S, u, v, x, y, z) if successful, else (None,)*6."""
        S_list = list(S_set)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S_set]
        random.shuffle(candidates)
        
        attempts = 0
        for i in range(len(S_list)):
            if attempts >= max_attempts:
                break
            u = S_list[i]
            for j in range(i + 1, len(S_list)):
                attempts += 1
                if attempts >= max_attempts:
                    break
                v = S_list[j]
                temp_S = S_set - {u, v}
                
                # Find all candidates that can be added to temp_S
                addable = []
                for x in candidates:
                    if is_valid_add(temp_S, x):
                        addable.append(x)
                
                # Try triples from addable
                for a in range(len(addable)):
                    for b in range(a + 1, len(addable)):
                        x = addable[a]
                        y = addable[b]
                        temp_S_with_xy = temp_S | {x, y}
                        for c in range(b + 1, len(addable)):
                            z = addable[c]
                            if is_valid_add(temp_S_with_xy, z):
                                new_S = temp_S | {x, y, z}
                                return new_S, u, v, x, y, z
        return None, None, None, None, None, None

    # VNS Loop
    stuck_count = 0
    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - start_time

        # N0: Add 1
        new_S, _ = try_add_one(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
            stuck_count = 0
            continue
        
        # N1: 1-for-2
        # Use more attempts when we have more time
        max_attempts = 500 if remaining > 30 else (300 if remaining > 15 else 100)
        new_S, _, _, _ = try_swap_1_for_2(S, max_attempts)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
            stuck_count = 0
            continue
        
        # N2: 2-for-3 (only if we have enough time)
        if remaining > 20:
            max_attempts = 200 if remaining > 40 else 100
            new_S, _, _, _, _, _ = try_swap_2_for_3(S, max_attempts)
            if new_S:
                S = new_S
                current_size = len(S)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                stuck_count = 0
                continue
        
        # If stuck, try a 1-for-1 swap to change the structure (size remains same)
        improved_structure = False
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates)
        
        for u in S_list:
            if time.time() > deadline:
                break
            temp_S = S - {u}
            for v in candidates:
                if is_valid_add(temp_S, v):
                    # This swap is valid
                    S = temp_S | {v}
                    improved_structure = True
                    break
            if improved_structure:
                break
        
        if improved_structure:
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > 30:
                # Random restart from best
                S = best_set.copy()
                stuck_count = 0
                # Maybe try a random 1-for-1 swap to perturb
                S_list = list(S)
                if S_list:
                    u = random.choice(S_list)
                    v = random.randint(1, N)
                    if v not in S:
                        temp_S = S - {u}
                        if is_valid_add(temp_S, v):
                            S = temp_S | {v}

    # Final polish: try to add more elements
    for x in range(1, N + 1):
        if x not in best_set:
            if is_valid_add(best_set, x):
                best_set.add(x)

    return list(best_set)
