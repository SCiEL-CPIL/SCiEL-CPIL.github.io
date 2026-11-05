"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_counts(A, n=128):
    """Compute representation counts for sums in [n, 2n] using unordered pairs."""
    A = np.asarray(A, dtype=np.int64)
    # We need counts for each s in [n, 2n] of the number of unordered pairs {a,b} in A with a<=b and a+b=s
    # Use convolution: conv[i] = sum_{a+b=i} 1 for all a,b in A (ordered)
    # For unordered with equal allowed: count_unordered(s) = (conv[s] + (1 if s even and s/2 in A else 0)) // 2
    # But careful: conv counts ordered pairs (a,b) where a,b in A.
    # For s, ordered count = sum_{a in A, b in A, a+b=s} 1
    # Unordered count with equal permitted: for s, if s is even and s//2 in A, then there's one pair (s//2, s//2)
    # and the rest come in pairs (a,b) and (b,a) with a!=b
    # So unordered_count = (ordered_count + (1 if s%2==0 and s//2 in A_set else 0)) // 2
    
    A_set = set(A.tolist())
    # Compute convolution
    # A is a subset of [0, n], so we can use FFT or direct
    # For n=128, A has at most 129 elements, convolution is fast
    a_arr = np.zeros(257, dtype=np.float64)
    for a in A:
        a_arr[a] = 1.0
    # Convolution
    conv = np.convolve(a_arr, a_arr, mode='full')
    # conv[s] gives ordered count for sum s
    counts = np.zeros(2*n + 1, dtype=np.int64)
    for s in range(n, 2*n + 1):
        ordered = int(round(conv[s]))
        has_diag = (s % 2 == 0) and ((s // 2) in A_set)
        unordered = (ordered + (1 if has_diag else 0)) // 2
        counts[s] = unordered
    return counts

def _evaluate(A):
    """Return (max_count, min_count, is_valid) for the set A."""
    counts = _compute_counts(A)
    # Check coverage: every integer in [n, 2n] must have at least one representation
    target = counts[128:257]  # sums 128 to 256
    if np.any(target == 0):
        return (float('inf'), 0, False)
    return (int(np.max(target)), int(np.min(target)), True)

def _random_init(n=128):
    """Generate a random initial set A."""
    # Start with a reasonable size
    size = np.random.randint(50, 100)
    A = set(np.random.choice(n + 1, size=size, replace=False).tolist())
    return A

def _repair(A, n=128):
    """Try to repair coverage by adding elements."""
    A = set(A)
    counts = _compute_counts(A)
    for s in range(n, 2*n + 1):
        if counts[s] == 0:
            # Need to add elements to cover s
            # Try adding pairs
            for a in range(0, s + 1):
                b = s - a
                if 0 <= b <= n:
                    if a not in A and b not in A:
                        # Add both? Or just one if the other is already there
                        pass
                    # Try adding a if b is in A, or b if a is in A
                    if b in A and a not in A:
                        A.add(a)
                        counts = _compute_counts(A)
                        break
                    elif a in A and b not in A:
                        A.add(b)
                        counts = _compute_counts(A)
                        break
                    elif a not in A and b not in A:
                        # Add one of them, say a
                        A.add(a)
                        counts = _compute_counts(A)
                        break
    return A

def _local_search(A, n=128, time_limit=50.0):
    """Local search to minimize peak multiplicity."""
    A = set(A)
    best_A = set(A)
    best_score = float('inf')
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        current_score, _, valid = _evaluate(A)
        if not valid:
            A = _repair(A, n)
            continue
        if current_score < best_score:
            best_score = current_score
            best_A = set(A)
        
        # Try moves: add, remove, swap
        improved = False
        # Try removals
        for a in list(A):
            A.remove(a)
            score, _, valid = _evaluate(A)
            if valid and score < current_score:
                current_score = score
                improved = True
                break
            else:
                A.add(a)
        
        if improved:
            continue
        
        # Try additions
        for a in range(n + 1):
            if a not in A:
                A.add(a)
                score, _, valid = _evaluate(A)
                if valid and score < current_score:
                    current_score = score
                    improved = True
                    A.discard(a)  # Don't add yet, just note it
                    A.add(a)
                    break
                else:
                    A.discard(a)
        
        if improved:
            continue
        
        # Try swaps
        for a in list(A):
            for b in range(n + 1):
                if b in A:
                    continue
                A.remove(a)
                A.add(b)
                score, _, valid = _evaluate(A)
                if valid and score < current_score:
                    current_score = score
                    improved = True
                    break
                else:
                    A.remove(b)
                    A.add(a)
            if improved:
                break
        
        if not improved:
            # Stuck, try a random perturbation
            if len(A) > 30:
                # Remove a random element
                a = np.random.choice(list(A))
                A.discard(a)
            # Add a random element
            b = np.random.randint(0, n + 1)
            A.add(b)
            A = _repair(A, n)
    
    return best_A, best_score

def construct():
    """Construct a subset A of [0,128] covering [128,256] by pair sums, minimizing peak multiplicity."""
    n = 128
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_A = None
    best_score = float('inf')
    
    # Start with the seed solution
    A0 = set(range(64, 129))
    score0, _, valid0 = _evaluate(A0)
    if valid0:
        best_A = set(A0)
        best_score = score0
    
    # Run local search from seed
    A_ls, score_ls = _local_search(A0, n, time_limit=40.0)
    if score_ls < best_score:
        best_A = set(A_ls)
        best_score = score_ls
    
    # Try a few random starts if time permits
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        A_rand = _random_init(n)
        A_rand = _repair(A_rand, n)
        A_rand_ls, score_rand = _local_search(A_rand, n, time_limit=8.0)
        if score_rand < best_score:
            best_A = set(A_rand_ls)
            best_score = score_rand
    
    if best_A is None:
        best_A = A0
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
