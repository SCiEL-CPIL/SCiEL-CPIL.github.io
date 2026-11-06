import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    n = 256
    
    # Fast can_add using bitmasks for each residue class
    # Precompute for each a in [1,256], which pairs (b,c) with b<=c have (b+c)%a==0
    # We'll use a different approach: for each a, store the set of values that conflict
    
    # For speed, precompute conflict sets
    # conflict[a] = set of x such that there exist b,c in S with a|b+c and x is involved
    # This is complex, so let's use a smarter greedy
    
    # Key insight: if we take all numbers in (n/2, n], no two sum to a multiple of any smaller number
    # because b+c > n >= 2*max(a) for a in the set... wait, a is also in the set.
    
    # Better approach: use the structure that large numbers are "safe"
    # For a > n/2, b+c >= 2*(a+1) > 2a, so a can only divide b+c if b+c >= 2a
    # But b,c > a, so b+c > 2a, meaning a | b+c requires b+c >= 2a which is always true
    # The issue is whether b+c is exactly a multiple of a.
    
    # Let's use a faster can_add
    def can_add_fast(S, x):
        S_list = sorted(S)
        # Case 1: x is smallest (a=x)
        # Need: for all b,c in S with b>x, c>=b, (b+c)%x != 0
        for i in range(len(S_list)):
            b = S_list[i]
            if b > x:
                for j in range(i, len(S_list)):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        # Case 2: x is middle (b=x)
        # Need: for all a,c in S with a<x, c>=x, (x+c)%a != 0
        for a in S_list:
            if a < x:
                for c in S_list:
                    if c >= x and (x + c) % a == 0:
                        return False
        # Case 3: x is largest (c=x)
        # Need: for all a,b in S with a<b<=x, (b+x)%a != 0
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if b <= x and (b + x) % a == 0:
                    return False
        return True
    
    best_set = set(range(192, 257))  # 65 elements
    
    # Greedy from 256 down to 1
    S = set()
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        if can_add_fast(S, x):
            S.add(x)
    
    if len(S) > len(best_set):
        best_set = S
    
    # Greedy from 1 up to 256
    S2 = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if can_add_fast(S2, x):
            S2.add(x)
    
    if len(S2) > len(best_set):
        best_set = S2
    
    # Local search: try removing one element and adding one or more
    S3 = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S3_list = sorted(S3)
        for i in range(len(S3_list)):
            if time.time() > deadline:
                break
            a_remove = S3_list[i]
            S3_trial = S3 - {a_remove}
            # Try to add multiple missing numbers
            added = 0
            for x in range(256, 0, -1):
                if x in S3_trial:
                    continue
                if can_add_fast(S3_trial, x):
                    S3_trial.add(x)
                    added += 1
                    if added >= 2:
                        break
            if len(S3_trial) > len(S3):
                S3 = S3_trial
                improved = True
                break
    
    if len(S3) > len(best_set):
        best_set = S3
    
    # Try random restarts with perturbation
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Start from a random subset of best_set
            S4 = set()
            S4_list = sorted(best_set)
            random.seed(42)
            for x in S4_list:
                if random.random() < 0.8:
                    S4.add(x)
            # Try to add more elements
            for x in range(256, 0, -1):
                if time.time() > deadline - 5:
                    break
                if x not in S4 and can_add_fast(S4, x):
                    S4.add(x)
            if len(S4) > len(best_set):
                best_set = S4
    
    return sorted(best_set)
