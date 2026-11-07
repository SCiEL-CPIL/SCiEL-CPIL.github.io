import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 128
    
    # Precompute all forbidden triples: (a,b,c) with a<b<c such that
    # if a,b,c are all in S, then we need a+b, a+c, b+c NOT all in S
    # Actually, the constraint is: no three DISTINCT a,b,c in S such that
    # a+b, a+c, b+c are ALL in S.
    # So a triple (a,b,c) is "forbidden" if having all of {a,b,c,a+b,a+c,b+c} in S is bad.
    # But we're choosing S, so we need to ensure we don't pick a,b,c where all three pair sums are also in S.
    
    # For efficiency, let's precompute for each pair (a,b) with a<b, the sum s=a+b.
    # If s <= N and s is in S, then we must NOT have any c in S (c!=a,b) such that a+c in S and b+c in S.
    
    # Strategy: 
    # 1. Start with upper half {65,...,128}
    # 2. Greedily try to add each number from 1 to 64, keeping it if valid
    # 3. Local search: try swaps (remove one, add another) to improve
    # 4. Multiple restarts with random perturbations
    
    def is_valid(S):
        """Check if set S is valid. O(N^3) but with early exit."""
        S_list = sorted(S)
        S_set = S
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                ab = a + b
                if ab > N or ab not in S_set:
                    continue
                # Found a pair with sum in S. Check for c.
                for k in range(j+1, n):
                    c = S_list[k]
                    ac = a + c
                    bc = b + c
                    if ac <= N and bc <= N and ac in S_set and bc in S_set:
                        return False
        return True
    
    # Faster check: for a given set S, check validity by iterating pairs
    def is_valid_fast(S):
        S_set = S
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                ab = a + b
                if ab > N:
                    break
                if ab not in S_set:
                    continue
                # Need to check if exists c in S, c != a, c != b, with a+c in S and b+c in S
                # c must be in S, c+a in S, c+b in S
                # Iterate through remaining elements
                for k in range(j+1, n):
                    c = S_list[k]
                    if a + c > N or b + c > N:
                        break
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True
    
    # Start with the known good solution
    base_set = set(range(N//2 + 1, N + 1))
    best_S = base_set.copy()
    best_size = len(best_S)
    
    # Phase 1: Greedy addition of small numbers
    current = base_set.copy()
    for x in range(1, N//2 + 1):
        if time.time() > deadline:
            break
        trial = current | {x}
        if is_valid_fast(trial):
            current = trial
    
    if len(current) > best_size:
        best_size = len(current)
        best_S = current.copy()
    
    # Phase 2: Local search - try to add any remaining number
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, N + 1):
                if x in current:
                    continue
                trial = current | {x}
                if is_valid_fast(trial):
                    current = trial
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_S = current.copy()
    
    # Phase 3: Local search with swaps
    if time.time() < deadline:
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Try random swap: remove one element, add another
            if len(current) > 0:
                x_remove = random.choice(list(current))
                x_add = random.choice([x for x in range(1, N + 1) if x not in current])
                
                trial = current.copy()
                trial.discard(x_remove)
                trial.add(x_add)
                
                if is_valid_fast(trial):
                    # Accept if same size or larger
                    if len(trial) >= len(current):
                        if len(trial) > len(current) or random.random() < 0.3:
                            current = trial
                            if len(current) > best_size:
                                best_size = len(current)
                                best_S = current.copy()
    
    # Phase 4: Perturbation - remove some elements, then greedily re-add
    if time.time() < deadline:
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            trial = current.copy()
            # Remove 1-3 random elements
            to_remove = random.sample(list(trial), min(3, len(trial)))
            for x in to_remove:
                trial.discard(x)
            
            # Greedily add back
            for x in range(1, N + 1):
                if x in trial:
                    continue
                t2 = trial | {x}
                if is_valid_fast(t2):
                    trial = t2
            
            if len(trial) > len(current):
                current = trial
                if len(current) > best_size:
                    best_size = len(current)
                    best_S = current.copy()
    
    # Final validation
    if is_valid_fast(best_S):
        return sorted(best_S)
    else:
        # Fallback to safe set
        return sorted(base_set)
