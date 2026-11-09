import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 110  # Safety margin
    
    # Data structures for current set S
    S = []
    in_S = [False] * (max_val + 1)
    counts = [0] * (2 * max_val + 1)
    
    def try_add(x):
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    def try_remove(x):
        if x not in S:
            return False
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1
        return True

    def is_valid_set(S_candidate):
        # Check if S_candidate is a valid B2[2] set
        # S_candidate is a list of unique ints
        if len(S_candidate) != len(set(S_candidate)):
            return False
        c = [0] * (2 * max_val + 1)
        for i in range(len(S_candidate)):
            a = S_candidate[i]
            c[2 * a] += 1
            if c[2 * a] > 2:
                return False
            for j in range(i + 1, len(S_candidate)):
                b = S_candidate[j]
                s = a + b
                c[s] += 1
                if c[s] > 2:
                    return False
        return True

    def greedy_random():
        S_local = []
        in_S_local = [False] * (max_val + 1)
        counts_local = [0] * (2 * max_val + 1)
        
        def try_add_local(x):
            if counts_local[2 * x] >= 2:
                return False
            for y in S_local:
                if y == x:
                    continue
                s = x + y
                if counts_local[s] >= 2:
                    return False
            S_local.append(x)
            in_S_local[x] = True
            counts_local[2 * x] += 1
            for y in S_local:
                if y != x:
                    counts_local[x + y] += 1
            return True

        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        for x in candidates:
            try_add_local(x)
        return S_local

    # 1. Find a good initial set using multiple randomized greedy passes
    best_S = []
    best_size = 0
    
    # Run a few quick greedy passes
    for _ in range(20):
        S_candidate = greedy_random()
        if len(S_candidate) > best_size:
            best_size = len(S_candidate)
            best_S = S_candidate[:]
            
    # Initialize global structures with best_S
    S = best_S[:]
    in_S = [False] * (max_val + 1)
    counts = [0] * (2 * max_val + 1)
    for x in S:
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
                
    # 2. Local Search
    # We will try to improve the set size.
    # Moves:
    # - 1-for-1 swap: remove a, add b. Size constant.
    # - 1-for-2 swap: remove a, add b, c. Size +1.
    # - 2-for-3 swap: remove a, b, add c, d, e. Size +1.
    
    # To speed up, we can pre-calculate "conflicts" or just check directly.
    # Given small N, direct check is fine.
    
    last_improvement = time.time()
    
    while time.time() < deadline:
        # Try 1-for-2 swaps (Grow)
        # Pick a random a in S
        a = random.choice(S)
        
        # Pick random b, c not in S
        b = random.randint(1, max_val)
        while in_S[b]:
            b = random.randint(1, max_val)
        c = random.randint(1, max_val)
        while in_S[c] or c == b:
            c = random.randint(1, max_val)
            
        # Check if S \ {a} U {b, c} is valid
        # Remove a
        try_remove(a)
        
        # Try add b
        if try_add(b):
            # Try add c
            if try_add(c):
                # Success
                last_improvement = time.time()
            else:
                # Rollback c
                try_remove(c)
                # Rollback b
                try_remove(b)
                # Restore a
                try_add(a)
        else:
            # Rollback a
            try_add(a)
            
        # Occasionally try 1-for-1 swaps to shuffle
        if random.random() < 0.1:
             a = random.choice(S)
             b = random.randint(1, max_val)
             while in_S[b]:
                 b = random.randint(1, max_val)
             
             try_remove(a)
             if try_add(b):
                 pass
             else:
                 try_remove(b)
                 try_add(a)

    return S
