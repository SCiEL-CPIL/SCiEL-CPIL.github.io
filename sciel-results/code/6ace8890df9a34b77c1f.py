"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def _count_progression_diffs(S):
    """Count distinct d > 0 such that a, a+d, a+2d all in S."""
    S_set = set(S)
    elems = sorted(S)
    n = len(elems)
    valid_ds = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = elems[j] - elems[i]
            if elems[j] + d in S_set:
                valid_ds.add(d)
    return len(valid_ds)

def _greedy_construct():
    """Greedy construction: add elements one by one to maximize progression diffs."""
    # Start with a base set and grow it
    # Use a heuristic: start with arithmetic progressions and perturb
    
    best_score = 0
    best_set = None
    
    deadline = time.time() + 110.0  # Safety margin
    
    # Strategy 1: Start with small APs and merge
    # Try different starting points and step sizes
    
    for start in range(1, 20):
        for base_step in range(1, 10):
            # Build a set greedily
            S = set()
            # Add some initial elements
            for k in range(32):
                val = start + k * base_step
                if 1 <= val <= 512:
                    S.add(val)
            
            if len(S) != 32:
                # Fill with random elements
                while len(S) < 32:
                    S.add(random.randint(1, 512))
                if len(S) > 32:
                    S = set(list(S)[:32])
            
            score = _count_progression_diffs(S)
            if score > best_score:
                best_score = score
                best_set = list(S)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 2: Local search / hill climbing
    if best_set is not None:
        S = list(best_set)
        S_set = set(S)
        current_score = _count_progression_diffs(S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping one element with a random one
            for _ in range(5000):
                if time.time() > deadline:
                    break
                i = random.randint(0, 31)
                old_val = S[i]
                new_val = random.randint(1, 512)
                if new_val in S_set:
                    continue
                S[i] = new_val
                S_set.discard(old_val)
                S_set.add(new_val)
                new_score = _count_progression_diffs(S)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    S[i] = old_val
                    S_set.discard(new_val)
                    S_set.add(old_val)
        
        best_set = S
        best_score = current_score
    
    # Strategy 3: Simulated annealing
    if time.time() < deadline:
        S = list(best_set) if best_set else list(range(1, 33))
        S_set = set(S)
        current_score = _count_progression_diffs(S)
        best_SA_score = current_score
        best_SA_set = list(S)
        
        T = 10.0
        cooling = 0.995
        
        while T > 0.01 and time.time() < deadline:
            i = random.randint(0, 31)
            old_val = S[i]
            new_val = random.randint(1, 512)
            if new_val in S_set:
                continue
            S[i] = new_val
            S_set.discard(old_val)
            S_set.add(new_val)
            new_score = _count_progression_diffs(S)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_score = new_score
                if current_score > best_SA_score:
                    best_SA_score = current_score
                    best_SA_set = list(S)
            else:
                S[i] = old_val
                S_set.discard(new_val)
                S_set.add(old_val)
            T *= cooling
        
        if best_SA_score > best_score:
            best_score = best_SA_score
            best_set = best_SA_set
    
    return best_set if best_set else list(range(1, 33))

def construct():
    """Return exactly 32 distinct integers in 1..512 maximizing progression diffs."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Try multiple random restarts with different strategies
    best_set = None
    best_score = 0
    
    # First, try a structured approach
    # Use a base of small numbers with some spread
    candidates = []
    
    # Candidate 1: Dense at start
    candidates.append(list(range(1, 33)))
    
    # Candidate 2: Spread out
    candidates.append(list(range(1, 513, 16)))[:32]
    while len(candidates[-1]) < 32:
        candidates[-1].append(random.randint(1, 512))
    candidates[-1] = list(set(candidates[-1]))[:32]
    while len(candidates[-1]) < 32:
        candidates[-1].append(random.randint(1, 512))
    
    # Candidate 3: Two APs
    c3 = set()
    for k in range(16):
        c3.add(1 + k * 2)
    for k in range(16):
        c3.add(33 + k * 2)
    candidates.append(list(c3))
    
    # Candidate 4: Random
    candidates.append(random.sample(range(1, 513), 32))
    
    for c in candidates:
        score = _count_progression_diffs(c)
        if score > best_score:
            best_score = score
            best_set = list(c)
    
    # Local search on best candidate
    if best_set and time.time() < deadline:
        S = list(best_set)
        S_set = set(S)
        current_score = best_score
        no_improve_count = 0
        
        while time.time() < deadline and no_improve_count < 2000:
            improved = False
            for _ in range(1000):
                if time.time() > deadline:
                    break
                i = random.randint(0, 31)
                old_val = S[i]
                new_val = random.randint(1, 512)
                if new_val in S_set:
                    continue
                S[i] = new_val
                S_set.discard(old_val)
                S_set.add(new_val)
                new_score = _count_progression_diffs(S)
                if new_score > current_score:
                    current_score = new_score
                    no_improve_count = 0
                    improved = True
                else:
                    S[i] = old_val
                    S_set.discard(new_val)
                    S_set.add(old_val)
            if not improved:
                no_improve_count += 1000
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(S)
    
    return best_set if best_set else list(range(1, 33))

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
