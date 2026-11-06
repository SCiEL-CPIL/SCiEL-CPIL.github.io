import time
import random
from itertools import combinations
from functools import lru_cache

def is_sidon(S):
    """Check if a list of integers is a Sidon set (unique pair sums including diagonals)."""
    sums = set()
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    S_sorted = sorted(S)
    # Check from largest possible down to smallest
    for k in range(n, 1, -1):
        for subset in combinations(S_sorted, k):
            if is_sidon(subset):
                return k
    return 2  # Any 2 distinct elements form a Sidon set

def count_conflicts(S):
    """Count the number of unordered pair-sum collisions in S.
    Used as a secondary metric to guide search when max_sidon_size is tied."""
    sums = {}
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    total = 0
    for c in sums.values():
        if c > 1:
            total += c * (c - 1) // 2
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Initial candidates
    # Strategy: Start with dense sets in small ranges, as they tend to have smaller max Sidon sizes
    initial_sets = []
    # 1. Random dense sets in [0, 20]
    for _ in range(15):
        initial_sets.append(sorted(random.sample(range(0, 21), 12)))
    # 2. Arithmetic progressions with small step
    for d in [1, 2, 3]:
        for a in range(0, 128 - 11*d + 1, 3):
            initial_sets.append(sorted([a + i*d for i in range(12)]))
    # 3. Random sets in [0, 25]
    for _ in range(15):
        initial_sets.append(sorted(random.sample(range(0, 26), 12)))
    # 4. Random sets in [0, 35]
    for _ in range(10):
        initial_sets.append(sorted(random.sample(range(0, 36), 12)))
        
    # Evaluate initial candidates
    best_set = None
    best_score = 13  # Max possible is 12, so 13 is safe init
    best_conflicts = -1
    
    for S0 in initial_sets:
        if time.time() > deadline:
            break
        score = max_sidon_size(S0)
        if score < best_score:
            best_score = score
            best_set = S0[:]
            best_conflicts = count_conflicts(S0)
        elif score == best_score:
            # Tie-break: prefer higher conflicts (harder set)
            c = count_conflicts(S0)
            if c > best_conflicts:
                best_conflicts = c
                best_set = S0[:]

    if best_set is None:
        # Fallback
        best_set = sorted(random.sample(range(0, 21), 12))
        best_score = max_sidon_size(best_set)
        best_conflicts = count_conflicts(best_set)

    S = best_set[:]
    current_score = best_score
    current_conflicts = best_conflicts
    
    universe = list(range(129))
    n = 12
    
    # Local Search
    no_improve_count = 0
    max_no_improve = 300  # Allow more exploration before restart
    
    while time.time() < deadline:
        # Pick a random element to swap out
        idx = random.randrange(n)
        old_val = S[idx]
        
        # Pick a random replacement from universe not in S
        # Bias towards dense regions to create more conflicts
        r = random.random()
        if r < 0.4:
            new_val = random.randint(0, 20)
        elif r < 0.7:
            new_val = random.randint(0, 30)
        elif r < 0.85:
            new_val = random.randint(0, 50)
        else:
            new_val = random.choice(universe)
            
        if new_val in S:
            continue
            
        # Perform swap
        S[idx] = new_val
        new_S = sorted(S)
        
        # Evaluate
        new_score = max_sidon_size(new_S)
        
        if new_score < current_score:
            # Improvement in primary metric
            current_score = new_score
            current_conflicts = count_conflicts(new_S)
            best_set = new_S[:]
            best_score = current_score
            best_conflicts = current_conflicts
            no_improve_count = 0
        elif new_score == current_score:
            # Tie in primary metric. Check secondary metric (conflicts)
            new_conflicts = count_conflicts(new_S)
            if new_conflicts > current_conflicts:
                # Better secondary metric
                current_conflicts = new_conflicts
                best_set = new_S[:]
                best_conflicts = current_conflicts
                no_improve_count = 0
            else:
                # Revert if no improvement in either metric
                S[idx] = old_val
                no_improve_count += 1
        else:
            # Worse primary metric. Revert.
            S[idx] = old_val
            no_improve_count += 1
        
        # Restart logic
        if no_improve_count > max_no_improve:
            # Restart from a random dense set
            restart_range = random.choice([20, 25, 30, 35])
            S = sorted(random.sample(range(0, restart_range + 1), 12))
            current_score = max_sidon_size(S)
            current_conflicts = count_conflicts(S)
            if current_score < best_score:
                best_score = current_score
                best_set = S[:]
                best_conflicts = current_conflicts
            elif current_score == best_score and current_conflicts > best_conflicts:
                best_conflicts = current_conflicts
                best_set = S[:]
            no_improve_count = 0

    return sorted(best_set)
