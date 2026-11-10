"""Rewritten program: randomized local search for B2[2] sets in 1..1000."""
import time
import random
import numpy as np

# We need every sum s to have at most 2 representations a+b with a<=b from the set.
# Strategy:
# 1. Start with a random dense set (~700 elements).
# 2. Compute violation counts (sums with >2 representations).
# 3. Iteratively remove elements that contribute most to violations, or swap them.
# 4. Use NumPy for fast convolution-based sum counting.

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    N = 1000
    rng = np.random.default_rng(42)
    
    # Helper: compute sum representation counts using convolution
    # For a set S, the number of representations of s as a+b with a<=b
    # can be computed via: 
    #   Let f be indicator vector of S (length N+1)
    #   Convolution c = f * f gives counts of a+b (a,b in S, ordered)
    #   For a < b: each pair counted twice in c[s]
    #   For a = b: counted once in c[s]
    #   So: reps(s) = (c[s] + diag[s]) // 2 where diag[s] = 1 if s/2 in S and s even
    #   Actually: c[s] = sum_{a+b=s, a,b in S} 1 (ordered pairs)
    #   reps(s) with a<=b: 
    #     if s is even and s/2 in S: (c[s]-1)//2 + 1 = (c[s]+1)//2
    #     else: c[s]//2
    
    def count_reps(S_set):
        """S_set: set of ints in 1..N. Returns dict sum->count of a<=b reps."""
        f = np.zeros(N + 1, dtype=np.int32)
        for x in S_set:
            f[x] = 1
        c = np.convolve(f, f, mode='full')  # length 2N+1, c[k] = sum f[i]*f[k-i]
        # c[s] for s in 2..2N
        reps = {}
        for s in range(2, 2 * N + 1):
            cnt = c[s]
            if s % 2 == 0:
                half = s // 2
                if 1 <= half <= N and half in S_set:
                    cnt = (cnt + 1) // 2
                else:
                    cnt = cnt // 2
            else:
                cnt = cnt // 2
            if cnt > 0:
                reps[s] = cnt
        return reps
    
    def violation_score(S_set):
        """Returns total excess: sum over s of max(0, reps(s) - 2)."""
        reps = count_reps(S_set)
        return sum(max(0, v - 2) for v in reps.values())
    
    # Phase 1: Greedy construction to get a valid starting set
    # Use the seed algorithm as a baseline
    chosen_greedy = []
    counts = {}
    for x in range(1, N + 1):
        new = [x + y for y in chosen_greedy] + [2 * x]
        trial = dict(counts)
        for s in new:
            trial[s] = trial.get(s, 0) + 1
        if all(trial[s] <= 2 for s in new):
            chosen_greedy.append(x)
            counts = trial
    
    best_set = set(chosen_greedy)
    best_size = len(best_set)
    
    # Phase 2: Randomized local search
    # Start from a random dense set, then iteratively improve
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random initialization: pick ~700 random elements
        target_size = 700 + rng.integers(-50, 50)
        candidates = rng.choice(range(1, N + 1), size=min(target_size, N), replace=False)
        S_set = set(candidates.tolist())
        
        # Remove elements that cause the most violations
        # Iterate: find the element whose removal reduces violations the most
        improved = True
        max_rounds = 50
        round_num = 0
        while improved and round_num < max_rounds and time.time() < deadline - 15:
            improved = False
            round_num += 1
            
            current_viol = violation_score(S_set)
            if current_viol == 0:
                if len(S_set) > best_size:
                    best_set = set(S_set)
                    best_size = len(best_set)
                break
            
            # Try removing each element, pick the one that reduces violations most
            best_elem = None
            best_reduced_viol = current_viol
            # Sample a subset of elements to try (for speed)
            if len(S_set) > 200:
                sample = rng.choice(list(S_set), size=200, replace=False)
            else:
                sample = list(S_set)
            
            for elem in sample:
                if time.time() > deadline - 15:
                    break
                S_test = S_set - {elem}
                v = violation_score(S_test)
                if v < best_reduced_viol:
                    best_reduced_viol = v
                    best_elem = elem
            
            if best_elem is not None and best_reduced_viol < current_viol:
                S_set.discard(best_elem)
                improved = True
        
        # Try to add elements back
        if time.time() < deadline - 15:
            all_elements = set(range(1, N + 1)) - S_set
            if all_elements:
                add_candidates = list(all_elements)
                rng.shuffle(add_candidates)
                # Try adding each candidate
                for elem in add_candidates:
                    if time.time() > deadline - 15:
                        break
                    S_test = S_set | {elem}
                    if violation_score(S_test) == 0:
                        S_set.add(elem)
        
        # Update best
        if violation_score(S_set) == 0 and len(S_set) > best_size:
            best_set = set(S_set)
            best_size = len(S_set)
    
    # Final: try greedy addition on the best set found
    if time.time() < deadline - 5:
        S_set = set(best_set)
        for x in range(1, N + 1):
            if time.time() > deadline - 5:
                break
            if x not in S_set:
                S_test = S_set | {x}
                if violation_score(S_test) == 0:
                    S_set.add(x)
        if len(S_set) > best_size:
            best_set = set(S_set)
            best_size = len(S_set)
    
    return sorted(best_set)
