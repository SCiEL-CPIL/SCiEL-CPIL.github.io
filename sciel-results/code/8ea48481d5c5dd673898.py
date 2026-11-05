import time
import random

def _diffs_set(s):
    """Return set of nonzero absolute differences for a sorted list."""
    if len(s) < 2:
        return set()
    d = set()
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            d.add(lst[j] - lst[i])
    return d

def _is_sidon(s):
    """Check if s is a Sidon set (all pair sums a+b for a<=b are unique)."""
    if len(s) < 2:
        return True
    seen = set()
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sm = lst[i] + lst[j]
            if sm in seen:
                return False
            seen.add(sm)
    return True

def _can_add_sidon(s, x, diff_set_s, diff_set_other):
    """
    Check if x can be added to Sidon set s while maintaining:
    1. s ∪ {x} is still Sidon
    2. New diffs don't conflict with diff_set_s (internal) or diff_set_other
    Returns (valid, new_diffs) where new_diffs is the set of new diffs added.
    """
    if x in s:
        return False, set()
    
    new_diffs = set()
    for a in s:
        d = abs(x - a)
        if d in diff_set_s or d in diff_set_other:
            return False, set()
        new_diffs.add(d)
    
    # Check for duplicate new diffs (would mean a1+a2 = 2x = x+x)
    if len(new_diffs) != len(s):
        return False, set()
    
    # Verify Sidon condition for s ∪ {x} using sum check
    # New sums: x + a for a in s, and x + x
    existing_sums = set()
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            existing_sums.add(lst[i] + lst[j])
    
    for a in s:
        if (x + a) in existing_sums:
            return False, set()
    if (x + x) in existing_sums:
        return False, set()
    
    return True, new_diffs

def _greedy_construct(candidates, n=256, max_size=64):
    """Greedy construction trying to add elements to A or B."""
    A = []
    B = []
    D_A = set()
    D_B = set()
    
    for x in candidates:
        # Try adding to A
        if len(A) < max_size:
            valid, new_diffs = _can_add_sidon(A, x, D_A, D_B)
            if valid:
                D_A = D_A | new_diffs
                A.append(x)
                continue
        # Try adding to B
        if len(B) < max_size:
            valid, new_diffs = _can_add_sidon(B, x, D_B, D_A)
            if valid:
                D_B = D_B | new_diffs
                B.append(x)
    
    return A, B

def _local_search(A, B, rng, n=256, max_size=64, deadline=None):
    """Local search: add, remove, move, swap operations."""
    A = list(A)
    B = list(B)
    D_A = _diffs_set(A)
    D_B = _diffs_set(B)
    
    def update_diffs(S, D, x):
        new_D = set(D)
        for a in S:
            new_D.add(abs(x - a))
        return new_D
    
    def try_add_to(S, D_S, D_other, x):
        """Try to add x to set S with diff sets D_S and D_other."""
        if len(S) >= max_size or x in S:
            return False
        valid, new_diffs = _can_add_sidon(S, x, D_S, D_other)
        if valid:
            return new_diffs
        return None
    
    while True:
        if deadline and time.time() > deadline:
            break
        
        improved = False
        action = rng.random()
        
        # 1. Add to A
        if action < 0.25 and len(A) < max_size:
            for _ in range(100):
                x = rng.randint(1, n)
                result = try_add_to(A, D_A, D_B, x)
                if result is not None:
                    D_A = D_A | result
                    A.append(x)
                    improved = True
                    break
        
        # 2. Add to B
        elif action < 0.5 and len(B) < max_size:
            for _ in range(100):
                x = rng.randint(1, n)
                result = try_add_to(B, D_B, D_A, x)
                if result is not None:
                    D_B = D_B | result
                    B.append(x)
                    improved = True
                    break
        
        # 3. Move A to B
        elif action < 0.75 and A and len(B) < max_size:
            idx = rng.randrange(len(A))
            x = A[idx]
            A_temp = A[:idx] + A[idx + 1:]
            D_A_temp = _diffs_set(A_temp)
            # Check if x can be added to B given D_A_temp as the "other" set
            valid, new_diffs = _can_add_sidon(B, x, D_B, D_A_temp)
            if valid:
                D_B = D_B | new_diffs
                A = A_temp
                D_A = D_A_temp
                B.append(x)
                improved = True
        
        # 4. Move B to A
        elif B and len(A) < max_size:
            idx = rng.randrange(len(B))
            x = B[idx]
            B_temp = B[:idx] + B[idx + 1:]
            D_B_temp = _diffs_set(B_temp)
            valid, new_diffs = _can_add_sidon(A, x, D_A, D_B_temp)
            if valid:
                D_A = D_A | new_diffs
                B = B_temp
                D_B = D_B_temp
                A.append(x)
                improved = True
        
        if not improved:
            # Try a swap: remove one from A and one from B, try to add them to the other
            if rng.random() < 0.3 and A and B:
                idx_a = rng.randrange(len(A))
                idx_b = rng.randrange(len(B))
                xa = A[idx_a]
                xb = B[idx_b]
                A_temp = A[:idx_a] + A[idx_a + 1:]
                B_temp = B[:idx_b] + B[idx_b + 1:]
                D_A_temp = _diffs_set(A_temp)
                D_B_temp = _diffs_set(B_temp)
                # Try adding xa to B_temp and xb to A_temp
                valid_a, new_diffs_a = _can_add_sidon(B_temp, xa, D_B_temp, D_A_temp)
                valid_b, new_diffs_b = _can_add_sidon(A_temp, xb, D_A_temp, D_B_temp)
                if valid_a and valid_b:
                    # Check that new_diffs_a doesn't conflict with D_A_temp ∪ new_diffs_b
                    # and new_diffs_b doesn't conflict with D_B_temp ∪ new_diffs_a
                    if not (new_diffs_a & (D_A_temp | new_diffs_b)) and \
                       not (new_diffs_b & (D_B_temp | new_diffs_a)):
                        D_B = D_B_temp | new_diffs_a
                        D_A = D_A_temp | new_diffs_b
                        B = B_temp + [xa]
                        A = A_temp + [xb]
                        improved = True
            
            if not improved and rng.random() < 0.15:
                break
    
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_score = 0
    best_A = []
    best_B = []
    
    n = 256
    max_size = 64
    
    # Multiple strategies with different orderings
    strategies = [
        "random",
        "sorted",
        "reverse",
        "center_out",
        "prime_like"
    ]
    
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(seed)
        
        for strat in strategies:
            if time.time() > deadline - 5:
                break
            
            if strat == "random":
                candidates = list(range(1, n + 1))
                rng.shuffle(candidates)
            elif strat == "sorted":
                candidates = list(range(1, n + 1))
            elif strat == "reverse":
                candidates = list(range(n, 0, -1))
            elif strat == "center_out":
                candidates = list(range(1, n + 1))
                mid = n // 2
                candidates.sort(key=lambda x: abs(x - mid))
            elif strat == "prime_like":
                # Bias toward numbers with fewer small factors
                candidates = list(range(1, n + 1))
                candidates.sort(key=lambda x: (x % 2, x % 3, x % 5))
            
            A, B = _greedy_construct(candidates, n, max_size)
            
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
                
            # Local search
            if time.time() < deadline - 3:
                A_ls, B_ls = _local_search(A, B, rng, n, max_size, deadline)
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
                    
    if not best_A or not best_B:
        best_A = [1]
        best_B = [2]
        
    return [best_A, best_B]
