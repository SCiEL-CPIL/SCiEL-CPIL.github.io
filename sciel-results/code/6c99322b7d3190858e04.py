import random
import time
from itertools import combinations

def _is_sidon(S):
    """Check if S is a Sidon set (all pair sums including diagonals are unique)."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(A):
    """Find the maximum size of a Sidon subset of A."""
    n = len(A)
    # Check sizes from n down to 1
    for k in range(n, 0, -1):
        for subset in combinations(A, k):
            if _is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    def _validate(A):
        if len(A) != 12:
            return False
        if len(set(A)) != 12:
            return False
        if any(x < 0 or x > 128 for x in A):
            return False
        return True

    best_set = None
    best_size = 12  # We want to minimize this

    def _try_set(A):
        nonlocal best_set, best_size
        if not _validate(A):
            return
        ms = _max_sidon_size(A)
        if ms < best_size:
            best_size = ms
            best_set = A
            if best_size <= 5:
                return True
        return False

    def _greedy_construct(rng):
        S = []
        sums = set()
        for _ in range(12):
            best_x = -1
            best_score = -1
            # Randomize order to break ties and explore different paths
            candidates = list(range(129))
            rng.shuffle(candidates)
            for x in candidates:
                if x in S:
                    continue
                # Calculate new sums
                new_sums = set(sums)
                new_sums.add(2 * x)
                for s in S:
                    new_sums.add(x + s)
                
                # Score: number of collisions
                # Max unique sums for size k is k(k+1)/2
                k = len(S) + 1
                max_unique = k * (k + 1) // 2
                actual_unique = len(new_sums)
                collisions = max_unique - actual_unique
                
                # Add a small random noise to avoid getting stuck in same local minima
                score = collisions + rng.uniform(0, 0.1)
                
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x != -1:
                S.append(best_x)
                sums.add(2 * best_x)
                for s in S[:-1]:
                    sums.add(best_x + s)
        
        return sorted(S)

    def _local_search(A, time_limit):
        current_set = A[:]
        current_size = _max_sidon_size(current_set)
        
        while time.time() < time_limit:
            improved = False
            # Try swapping each element
            for i in range(12):
                old_val = current_set[i]
                # Try a subset of candidates to save time, or all if fast
                # 129 candidates * 12 positions = 1548 evals
                # Each eval ~0.5ms -> 0.8s per sweep.
                # We have ~80s, so ~100 sweeps.
                
                # To speed up, we can prioritize candidates that are close to old_val
                # or just iterate all.
                for new_val in range(129):
                    if new_val == old_val or new_val in current_set:
                        continue
                    
                    new_set = current_set[:]
                    new_set[i] = new_val
                    new_set.sort()
                    
                    # Quick check: if new_set is not valid, skip
                    # (Validation is cheap)
                    if not _validate(new_set):
                        continue
                    
                    new_size = _max_sidon_size(new_set)
                    
                    if new_size < current_size:
                        current_set = new_set
                        current_size = new_size
                        improved = True
                        # Restart the loop to re-evaluate with new set
                        break
                
                if improved:
                    break
                
                if time.time() > time_limit:
                    return current_set, current_size

            if not improved:
                # If no improvement found in a full sweep, try some random perturbations
                # to escape local minima
                for _ in range(50):
                    if time.time() > time_limit:
                        break
                    idx = random.randint(0, 11)
                    old_val = current_set[idx]
                    new_val = random.randint(0, 128)
                    if new_val == old_val or new_val in current_set:
                        continue
                    
                    new_set = current_set[:]
                    new_set[idx] = new_val
                    new_set.sort()
                    
                    if not _validate(new_set):
                        continue
                    
                    new_size = _max_sidon_size(new_set)
                    
                    # Accept if better, or with small probability if equal (annealing)
                    # But we want to minimize, so only accept if better or equal (to wander)
                    if new_size < current_size:
                        current_set = new_set
                        current_size = new_size
                        improved = True
                        break
                    elif new_size == current_size and random.random() < 0.1:
                        # Wander
                        current_set = new_set
                        # current_size remains same
                        # Don't mark as improved to allow more wandering?
                        # Actually, let's just break and restart sweep if we wandered?
                        # No, let's continue.
        
        return current_set, current_size

    # Phase 1: Greedy Construction
    rng = random.Random(42)
    initial_candidates = []
    
    # Run multiple greedy instances
    for i in range(50):
        if time.time() > deadline - 20.0:
            break
        A = _greedy_construct(rng)
        initial_candidates.append(A)
    
    # Also add some structured candidates
    # Small intervals
    for start in range(0, 10):
        A = list(range(start, start + 12))
        initial_candidates.append(A)
        
    # Arithmetic progressions
    for gap in range(2, 10):
        for start in range(0, 129 - 11*gap):
            A = [start + k*gap for k in range(12)]
            if A[-1] <= 128:
                initial_candidates.append(A)
            if time.time() > deadline - 20.0:
                break

    # Evaluate initial candidates
    for A in initial_candidates:
        if time.time() > deadline - 10.0:
            break
        _try_set(A)

    # Phase 2: Local Search on best candidates
    # Sort initial candidates by their max sidon size
    evaluated_candidates = []
    for A in initial_candidates:
        if not _validate(A):
            continue
        sz = _max_sidon_size(A)
        evaluated_candidates.append((sz, A))
    
    evaluated_candidates.sort(key=lambda x: x[0])
    
    # Take top 5 distinct sets for local search
    unique_sets = []
    for sz, A in evaluated_candidates:
        if A not in unique_sets:
            unique_sets.append(A)
        if len(unique_sets) >= 5:
            break
    
    for A in unique_sets:
        if time.time() > deadline - 5.0:
            break
        new_set, new_size = _local_search(A, deadline - 5.0)
        if new_size < best_size:
            best_size = new_size
            best_set = new_set
        elif new_size == best_size and best_set is not None:
            # Prefer sets with more collisions? Not necessary, just keep best.
            pass

    # Fallback
    if best_set is None:
        best_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    if not _validate(best_set):
        best_set = [3*i for i in range(12)]
        
    return best_set
