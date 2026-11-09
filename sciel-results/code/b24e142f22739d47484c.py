import random
import itertools
import time

def is_sidon(S):
    """Check if set S is a Sidon set (all unordered pair sums unique including diagonals)."""
    sums = set()
    for i, x in enumerate(S):
        for j, y in enumerate(S):
            if i <= j:
                s = x + y
                if s in sums:
                    return False
                sums.add(s)
    return True

def has_sidon_subset_of_size_k(A, k):
    """Check if A contains a Sidon subset of size k."""
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def _max_sidon_size(A):
    """Find the maximum size of a Sidon subset in A."""
    for k in range(len(A), 0, -1):
        if has_sidon_subset_of_size_k(A, k):
            return k
    return 0

def _collision_score(A):
    """Score a set by how many sum collisions it has (higher = more collisions = better for minimizing Sidon)."""
    sums = {}
    for i, x in enumerate(A):
        for j, y in enumerate(A):
            if i <= j:
                s = x + y
                sums[s] = sums.get(s, 0) + 1
    collisions = sum(v - 1 for v in sums.values() if v > 1)
    return collisions

def _greedy_build(start_set, pool, target_size=12):
    """Greedily build a set by adding elements that maximize collision density."""
    current = list(start_set)
    remaining = [x for x in pool if x not in current]
    
    while len(current) < target_size and remaining:
        best_elem = None
        best_score = -1
        for elem in remaining:
            test_set = current + [elem]
            score = _collision_score(test_set)
            if score > best_score:
                best_score = score
                best_elem = elem
        if best_elem is not None:
            current.append(best_elem)
            remaining.remove(best_elem)
        else:
            break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave margin for evaluator overhead
    
    # Baseline: range(12) has max Sidon size 5
    best_set = list(range(12))
    best_max_size = _max_sidon_size(best_set)
    
    # Phase 1: Greedy construction from random seeds
    pool = list(range(129))
    for _ in range(500):
        if time.time() > deadline:
            break
        
        # Start with a random 4-element seed
        seed = random.sample(pool, 4)
        A = _greedy_build(seed, pool)
        
        if len(A) == 12:
            max_size = _max_sidon_size(A)
            if max_size < best_max_size:
                best_set = A
                best_max_size = max_size
                if max_size <= 3:
                    return A
    
    # Phase 2: Local search - perturb best_set
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        A = list(best_set)
        # Randomly swap one or two elements
        num_swaps = random.randint(1, 2)
        for _ in range(num_swaps):
            idx = random.randint(0, 11)
            new_val = random.choice([x for x in range(129) if x not in A])
            A[idx] = new_val
        
        if len(set(A)) != 12:
            continue
        
        max_size = _max_sidon_size(A)
        if max_size < best_max_size:
            best_set = A
            best_max_size = max_size
            if max_size <= 3:
                return A
    
    return best_set
