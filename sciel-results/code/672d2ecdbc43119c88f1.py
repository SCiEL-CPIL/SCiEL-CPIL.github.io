import time
import random
import sys

def _diffs_set(s):
    """Return set of nonzero absolute differences."""
    if len(s) < 2:
        return set()
    d = set()
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            d.add(lst[j] - lst[i])
    return d

def _can_add(S, x, D_S, D_other):
    """
    Check if x can be added to S.
    S is Sidon iff all diffs in S are unique.
    So adding x is valid if:
    1. All new diffs |x-a| are not in D_S (maintains Sidon).
    2. All new diffs |x-a| are not in D_other (maintains disjoint diffs).
    """
    if x in S:
        return False
    
    new_diffs = set()
    for a in S:
        d = abs(x - a)
        if d in D_S or d in D_other:
            return False
        new_diffs.add(d)
        
    # Check for duplicate new diffs? 
    # If |x-a1| == |x-a2|, then a1 and a2 are symmetric around x.
    # Then a1 + a2 = 2x.
    # Also x+x = 2x.
    # So sum x+x collides with a1+a2.
    # This violates Sidon.
    # So we must ensure all new diffs are unique among themselves?
    # Yes, if d1 == d2, then x-a1 = x-a2 => a1=a2 (impossible) or x-a1 = a2-x => a1+a2=2x.
    # So we need len(new_diffs) == len(S).
    if len(new_diffs) != len(S):
        return False
        
    return True

def _greedy_construct(candidates, rng, n=256, max_size=64):
    """Greedy construction with diff-based checks."""
    A = []
    B = []
    D_A = set()
    D_B = set()
    
    for x in candidates:
        # Try A
        if len(A) < max_size:
            if _can_add(A, x, D_A, D_B):
                for a in A:
                    D_A.add(abs(x - a))
                A.append(x)
                continue
        # Try B
        if len(B) < max_size:
            if _can_add(B, x, D_B, D_A):
                for b in B:
                    D_B.add(abs(x - b))
                B.append(x)
    
    return A, B

def _local_search(A, B, rng, n=256, max_size=64, deadline=None):
    """Local search with diff-based checks."""
    A = list(A)
    B = list(B)
    D_A = _diffs_set(A)
    D_B = _diffs_set(B)
    
    # Helper to update diffs
    def update_diffs(S, D, x):
        new_D = set(D)
        for a in S:
            new_D.add(abs(x - a))
        return new_D

    while True:
        if deadline and time.time() > deadline:
            break
        
        improved = False
        # Randomly choose an action
        action = rng.random()
        
        # 1. Add to A
        if action < 0.3 and len(A) < max_size:
            for _ in range(50):
                x = rng.randint(1, n)
                if x not in A and _can_add(A, x, D_A, D_B):
                    D_A = update_diffs(A, D_A, x)
                    A.append(x)
                    improved = True
                    break
        
        # 2. Add to B
        elif action < 0.6 and len(B) < max_size:
            for _ in range(50):
                x = rng.randint(1, n)
                if x not in B and _can_add(B, x, D_B, D_A):
                    D_B = update_diffs(B, D_B, x)
                    B.append(x)
                    improved = True
                    break
        
        # 3. Move A to B
        elif action < 0.8 and A and len(B) < max_size:
            idx = rng.randrange(len(A))
            x = A[idx]
            A_temp = A[:idx] + A[idx+1:]
            D_A_temp = _diffs_set(A_temp)
            if _can_add(B, x, D_B, D_A_temp):
                # Check if D_A_temp and D_B are disjoint? 
                # _can_add checks D_B and D_A_temp.
                # But we also need D_A_temp and D_B to be disjoint?
                # Yes, _can_add(B, x, D_B, D_A_temp) ensures new diffs of x vs B are not in D_A_temp.
                # But does it ensure D_B and D_A_temp are disjoint?
                # No, D_B and D_A_temp were disjoint before (invariant).
                # Removing x from A might create new conflicts? No, removing elements only removes diffs.
                # So D_A_temp is a subset of old D_A (minus diffs involving x).
                # Old D_A and D_B were disjoint.
                # So D_A_temp and D_B are disjoint.
                # And _can_add ensures new diffs (x vs B) are not in D_A_temp.
                # So it's valid.
                D_B_new = update_diffs(B, D_B, x)
                A = A_temp
                D_A = D_A_temp
                B.append(x)
                D_B = D_B_new
                improved = True
        
        # 4. Move B to A
        elif B and len(A) < max_size:
            idx = rng.randrange(len(B))
            x = B[idx]
            B_temp = B[:idx] + B[idx+1:]
            D_B_temp = _diffs_set(B_temp)
            if _can_add(A, x, D_A, D_B_temp):
                D_A_new = update_diffs(A, D_A, x)
                B = B_temp
                D_B = D_B_temp
                A.append(x)
                D_A = D_A_new
                improved = True
        
        if not improved:
            # Stuck, break or try harder?
            if rng.random() < 0.1:
                break

    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_score = 0
    best_A = []
    best_B = []
    
    n = 256
    max_size = 64 # Theoretical max is 16, but keep 64 to be safe if logic changes
    
    # Multiple strategies
    strategies = [
        "random",
        "sorted",
        "reverse",
        "center_out"
    ]
    
    for seed in range(100):
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
                
            A, B = _greedy_construct(candidates, rng, n, max_size)
            
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
                
            # Local Search
            if time.time() < deadline - 2:
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
