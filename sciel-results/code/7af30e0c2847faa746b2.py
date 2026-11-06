import time
import random
from itertools import combinations
from collections import Counter

def _max_sidon_subset_size(candidates):
    """Exhaustively compute the largest subset of candidates where all unordered
    pair sums (including diagonals) are unique."""
    n = len(candidates)
    # We need to find the largest subset S such that for all pairs (i,j) with i<=j
    # (indices into S), the sums candidates[S[i]] + candidates[S[j]] are all distinct.
    # This is equivalent to: the multiset {a+b for a in S for b in S where a<=b} has
    # all distinct values.
    
    # For small subsets (up to ~12), we can use a branch-and-bound or bitmask approach.
    # Since n=12, we can try all subsets of size k from large to small.
    
    # Use bitmasks for subsets
    best_size = 0
    # Try all subsets of size from n down to 1
    # For n=12, total subsets = 2^12 = 4096, very manageable
    
    for mask in range(1, 1 << n):
        size = bin(mask).count('1')
        if size <= best_size:
            continue
        # Check if this subset is a valid Sidon set (with diagonals)
        subset = [candidates[i] for i in range(n) if mask & (1 << i)]
        m = len(subset)
        # All pair sums including diagonals
        sums = []
        for i in range(m):
            for j in range(i, m):
                sums.append(subset[i] + subset[j])
        if len(sums) == len(set(sums)):
            best_size = size
            if best_size == n:
                return best_size
    
    return best_size


def _is_valid_sidon(candidates, mask, n):
    """Check if the subset indicated by mask is a valid Sidon set."""
    subset = [candidates[i] for i in range(n) if mask & (1 << i)]
    m = len(subset)
    sums = set()
    for i in range(m):
        for j in range(i, m):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    random.seed(42)
    
    best_candidates = list(range(12))
    best_sidon_size = _max_sidon_subset_size(best_candidates)
    
    # Strategy: Use local search to find 12 integers in [0,128] that minimize
    # the largest Sidon subset size.
    # Key idea: create many sum collisions so that no large subset can avoid them.
    
    def _evaluate(candidates):
        return _max_sidon_subset_size(candidates)
    
    # Generate diverse initial candidates
    candidates_pool = []
    for _ in range(50):
        c = sorted(random.sample(range(0, 129), 12))
        candidates_pool.append(c)
    
    # Also try structured candidates
    # Arithmetic progressions create lots of sum collisions
    for d in range(2, 20):
        for start in range(0, 129 - 11 * d):
            c = [start + i * d for i in range(12)]
            if all(0 <= x <= 128 for x in c):
                candidates_pool.append(c)
    
    # Quadratic-like sequences
    for a in range(1, 15):
        for b in range(0, 10):
            c = sorted([a * i * i + b * i for i in range(12)])
            if max(c) <= 128:
                candidates_pool.append(c)
    
    # Evaluate pool
    for c in candidates_pool:
        if time.time() > deadline - 5:
            break
        if len(set(c)) != 12:
            continue
        if not all(0 <= x <= 128 for x in c):
            continue
        ss = _evaluate(c)
        if ss < best_sidon_size:
            best_sidon_size = ss
            best_candidates = list(c)
    
    # Local search: mutate one element at a time
    current = list(best_candidates)
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        improved = False
        # Try random mutations
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            trial = list(current)
            idx = random.randint(0, 11)
            old_val = trial[idx]
            # Try a new value
            new_val = random.randint(0, 128)
            while new_val in trial and new_val != old_val:
                new_val = random.randint(0, 128)
            trial[idx] = new_val
            if len(set(trial)) != 12:
                continue
            ss = _evaluate(trial)
            if ss < best_sidon_size:
                best_sidon_size = ss
                best_candidates = list(trial)
                current = list(trial)
                improved = True
                no_improve_count = 0
                break
            elif ss == best_sidon_size:
                # Accept equal to explore
                if random.random() < 0.1:
                    current = list(trial)
        
        if not improved:
            no_improve_count += 1
    
    # Final local polish: try all single-element swaps
    if time.time() < deadline - 3:
        current = list(best_candidates)
        for idx in range(12):
            old_val = current[idx]
            for new_val in range(0, 129):
                if new_val == old_val:
                    continue
                trial = list(current)
                trial[idx] = new_val
                if len(set(trial)) != 12:
                    continue
                ss = _evaluate(trial)
                if ss < best_sidon_size:
                    best_sidon_size = ss
                    best_candidates = list(trial)
                    current = list(trial)
                    break
            if time.time() > deadline - 3:
                break
    
    return list(best_candidates)


if __name__ == "__main__":
    result = construct()
    print(result)
    print("Max Sidon subset size:", _max_sidon_subset_size(result))
