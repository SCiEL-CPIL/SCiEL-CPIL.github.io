import time
import random
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_subset_size(S):
    S = sorted(S)
    n = len(S)
    # Try from largest k down
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_set = None
    best_val = 99
    
    # Initial candidates
    candidates = []
    
    # 1. Arithmetic Progressions
    for start in range(0, 120):
        for step in range(1, 10):
            S = [start + i * step for i in range(12)]
            if max(S) <= 128:
                candidates.append(S)
    
    # 2. Random sets
    random.seed(42)
    for _ in range(1000):
        S = random.sample(range(129), 12)
        candidates.append(S)
        
    # 3. Clustered sets (tight groups)
    for _ in range(1000):
        base = random.randint(0, 110)
        # Try to make a tight cluster
        offsets = random.sample(range(0, 20), 12)
        S = [base + o for o in offsets]
        if max(S) <= 128 and len(set(S)) == 12:
             candidates.append(S)

    # Evaluate initial candidates
    for S in candidates:
        if time.time() > deadline:
            break
        val = largest_sidon_subset_size(S)
        if val < best_val:
            best_val = val
            best_set = S[:]
            
    # Local Search
    if best_set is None:
        best_set = random.sample(range(129), 12)
        best_val = largest_sidon_subset_size(best_set)
        
    # Optimization Loop
    # We want to minimize best_val.
    # If we reach 3, that's likely optimal. 4 is also very good.
    
    current_set = best_set[:]
    current_val = best_val
    
    no_improvement_count = 0
    max_no_improvement = 5000
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        # Pick a random index to change
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Try a new value
        # Bias towards values close to current set to maintain structure?
        # Or random?
        new_val = random.randint(0, 128)
        
        if new_val == old_val or new_val in current_set:
            continue
            
        trial_set = current_set[:]
        trial_set[idx] = new_val
        
        val = largest_sidon_subset_size(trial_set)
        
        if val < current_val:
            current_set = trial_set
            current_val = val
            no_improvement_count = 0
            if current_val < best_val:
                best_val = current_val
                best_set = current_set[:]
                
            # If we hit 3, we can stop or try to find 2 (impossible? min is 1? No, min Sidon is 1. But we want to MINIMIZE the MAX. So 3 is very low.)
            # Actually, can we get 2?
            # If max Sidon size is 2, then no 3 elements form a Sidon set.
            # A 3-element set $\{a,b,c\}$ is Sidon if $2a \neq b+c, 2b \neq a+c, 2c \neq a+b$ and $a+b \neq a+c$ (implies $b \neq c$) etc.
            # Basically, no 3 elements in AP.
            # Can we have 12 integers with no 3 elements in AP?
            # This is a "3-free" set (no 3-term AP).
            # Max size of a 3-free set in $[0, 128]$?
            # Cap set problem / Roth's theorem.
            # For $N=129$, max size of 3-free set is roughly $N / \log \log N$?
            # Actually, for small N, it's known.
            # If we can find a 12-element 3-free set, then max Sidon size is 2?
            # Wait. If no 3 elements are in AP, does that mean no 3-element Sidon set?
            # A 3-element set is NOT Sidon if it has a collision.
            # Collisions for 3 elements $\{a,b,c\}$:
            # $2a = b+c$ (AP)
            # $a+b = a+c \implies b=c$ (distinct, so no)
            # $a+b = 2c$? No, sums are $2a, 2b, 2c, a+b, a+c, b+c$.
            # Collision if $2a = b+c$ OR $a+b = a+c$ (impossible) OR $a+b = 2c$?
            # Wait, $a+b = 2c$ is not a collision of pair sums unless $2c$ is also a pair sum.
            # Pair sums are $2a, 2b, 2c, a+b, a+c, b+c$.
            # Collision if any two are equal.
            # $2a = b+c$ (AP centered at a? No, $b+c=2a$ means a is midpoint. So $b,a,c$ is AP).
            # $a+b = a+c \implies b=c$.
            # $a+b = 2c$? This is a collision between sum $a+b$ and sum $2c$.
            # So if $a+b=2c$, it's not Sidon.
            # So "No 3-term AP" is NOT sufficient to ensure "No 3-element Sidon set".
            # We also need to avoid $a+b=2c$ type collisions.
            # So the condition for Max Sidon Size = 2 is:
            # No 3-element subset is Sidon.
            # This means EVERY 3-element subset has a collision.
            # This is a very strong condition.
            
            # So 3 is a safer target than 2.
            if current_val <= 3:
                # Try to polish for a bit
                pass
        else:
            no_improvement_count += 1
            
        # Occasionally restart from best
        if no_improvement_count > 500:
            current_set = best_set[:]
            current_val = best_val
            no_improvement_count = 0
            # Perturb current_set
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in current_set:
                    current_set[idx] = new_val
            
    return best_set
