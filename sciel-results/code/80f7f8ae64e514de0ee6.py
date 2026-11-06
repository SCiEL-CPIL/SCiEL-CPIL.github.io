import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    
    Structural approach: Model as an assignment problem. Each difference d in 1..30 must be
    realized by some pair (a, b) with b - a = d. Since 0 is in the set, d can be realized by
    (0, d) or (x, x+d) for some x > 0. We need to choose, for each d, a "base" element a_d
    such that all pairs (a_d, a_d + d) are consistent (same element appears in multiple pairs)
    and no difference is repeated.
    
    Alternative structural view: We're looking for a set S = {0} ∪ X where:
    - All differences |s_i - s_j| for s_i, s_j in S are distinct
    - Every d in 1..30 appears as some |s_i - s_j|
    
    This is a Sidon set. The key structural insight: if we fix the "shape" (which elements
    are in X), the differences are determined. We can model this as:
    
    For a given candidate set of elements, check Sidon + coverage.
    
    New approach: Use the fact that small Sidon sets are well-studied. For n elements,
    the minimal maximum element is related to perfect difference sets. For 31 elements
    (0 + 30 others), we need a (31, ?, ?) difference set.
    
    Actually, let me use a different structural approach: **integer programming via
    branch-and-bound with strong pruning**, but structured around the differences.
    
    Key insight: If we have a Sidon set S with 0, then for each d in 1..30, exactly one
    pair (a, a+d) exists in S. This means we can think of it as: for each d, choose a
    value a_d such that a_d and a_d + d are both in S, and all such pairs are consistent.
    
    This is a constraint satisfaction problem where variables are the elements of S\{0},
    and constraints are: (1) all differences distinct, (2) all d in 1..30 covered.
    
    Let me use a **genetic algorithm with fitness = -max(S) if valid, else large penalty**.
    The representation: a sorted list of integers including 0.
    """
    start_time = time.time()
    DEADLINE = 145.0
    
    TARGET = set(range(1, 31))
    
    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs
    
    def is_sidon(S):
        S = sorted(S)
        seen = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def covers_all(S):
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)
    
    def evaluate(S):
        """Return (is_valid, max_val) where is_valid means Sidon + covers all."""
        if not is_sidon(S):
            return (False, float('inf'))
        if not covers_all(S):
            return (False, float('inf'))
        return (True, max(S))
    
    # Known good starting point from literature / small search
    # A Sidon set with 0 and differences 1..30
    # Try the quadratic construction: S = {i^2 mod p} for appropriate p
    # Or use known small Sidon sets
    
    # Let's start with a heuristic construction
    def greedy_construct():
        """Greedy: build S element by element, always choosing the smallest
        element that covers the most missing differences."""
        S = [0]
        covered = set()
        used_diffs = set()
        
        for step in range(31):
            if time.time() > DEADLINE - 20:
                break
            if covered == TARGET:
                break
            
            best_x = None
            best_gain = -1
            best_new_diffs = None
            
            # Generate candidates: for each missing d and each s in S, try s+d and s-d
            candidates = set()
            missing = TARGET - covered
            for s in S:
                for d in missing:
                    candidates.add(s + d)
                    if s - d > 0:
                        candidates.add(s - d)
            
            # Also try small random values
            for _ in range(200):
                candidates.add(random.randint(1, 200))
            
            candidates = [c for c in candidates if c > 0 and c not in S and c <= 500]
            candidates.sort()
            
            for x in candidates:
                # Check if adding x maintains Sidon property
                valid = True
                new_diffs = set()
                for s in S:
                    d = abs(x - s)
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                
                gain = len(new_diffs & missing)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_new_diffs = new_diffs
            
            if best_x is None:
                # No candidate works; try a larger value
                for x in range(1, 1000):
                    if x in S:
                        continue
                    valid = True
                    new_diffs = set()
                    for s in S:
                        d = abs(x - s)
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid and (new_diffs & missing):
                        best_x = x
                        best_new_diffs = new_diffs
                        break
                if best_x is None:
                    break
            
            S.append(best_x)
            used_diffs |= best_new_diffs
            covered = used_diffs & TARGET
        
        return S
    
    # Local search: try to reduce max(S) by replacing elements
    def local_search(S, time_limit=30.0):
        S = sorted(S)
        best = S[:]
        best_max = max(S)
        t0 = time.time()
        
        while time.time() - t0 < time_limit:
            improved = False
            
            # Try replacing each non-zero element with a smaller value
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                old_val = S[i]
                
                # Try values from 1 to old_val - 1
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    
                    if not is_sidon(trial):
                        continue
                    if not covers_all(trial):
                        continue
                    
                    if max(trial) < best_max:
                        S = trial
                        best = S[:]
                        best_max = max(S)
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try swapping two elements
                for i in range(len(S)):
                    if S[i] == 0:
                        continue
                    for j in range(i + 1, len(S)):
                        if S[j] == 0:
                            continue
                        trial = S[:]
                        trial[i], trial[j] = trial[j], trial[i]
                        trial.sort()
                        if not is_sidon(trial):
                            continue
                        if not covers_all(trial):
                            continue
                        if max(trial) < best_max:
                            S = trial
                            best = S[:]
                            best_max = max(S)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best
    
    # Main: try multiple strategies
    best_S = None
    best_max = float('inf')
    
    # Strategy 1: Greedy construction
    random.seed(42)
    S1 = greedy_construct()
    if S1 and is_sidon(S1) and covers_all(S1):
        m = max(S1)
        if m < best_max:
            best_max = m
            best_S = S1
    
    # Strategy 2: Local search from greedy
    if best_S and time.time() < DEADLINE - 40:
        S2 = local_search(best_S, time_limit=30.0)
        if is_sidon(S2) and covers_all(S2):
            m = max(S2)
            if m < best_max:
                best_max = m
                best_S = S2
    
    # Strategy 3: Try different greedy seeds
    for seed in range(10):
        if time.time() > DEADLINE - 30:
            break
        random.seed(seed * 123 + 456)
        S = greedy_construct()
        if S and is_sidon(S) and covers_all(S):
            m = max(S)
            if m < best_max:
                best_max = m
                best_S = S
                # Quick local search
                if time.time() < DEADLINE - 20:
                    S_ref = local_search(S, time_limit=10.0)
                    if is_sidon(S_ref) and covers_all(S_ref):
                        m_ref = max(S_ref)
                        if m_ref < best_max:
                            best_max = m_ref
                            best_S = S_ref
    
    # Strategy 4: Simulated annealing on the set representation
    if best_S and time.time() < DEADLINE - 10:
        S = best_S[:]
        current_max = max(S)
        best_sa = S[:]
        best_sa_max = current_max
        T = 50.0
        t0 = time.time()
        
        while time.time() - t0 < 10.0:
            # Perturbation: replace one random non-zero element
            idx = random.choice([i for i in range(len(S)) if S[i] != 0])
            old_val = S[idx]
            new_val = random.randint(1, max(100, int(current_max * 1.2)))
            if new_val in S:
                continue
            S[idx] = new_val
            S.sort()
            
            if is_sidon(S) and covers_all(S):
                new_max = max(S)
                delta = new_max - current_max
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_max = new_max
                    if new_max < best_sa_max:
                        best_sa_max = new_max
                        best_sa = S[:]
            else:
                # Revert
                S[idx] = old_val
                S.sort()
            
            T *= 0.995
        
        if best_sa_max < best_max:
            best_max = best_sa_max
            best_S = best_sa
    
    # Final answer
    if best_S is None:
        # Fallback: should not happen
        best_S = [0]
        for d in range(1, 31):
            for x in range(1, 10000):
                trial = sorted(best_S + [x])
                if is_sidon(trial):
                    best_S = trial
                    break
    
    return best_S
