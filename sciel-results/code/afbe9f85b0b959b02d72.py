import time
import random
import itertools

def _compute_objective(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _compute_sums_prods(A):
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def _geometric_init(base, ratio, n=20, max_val=10000):
    """Generate geometric progression, ensuring distinct and <= max_val."""
    A = []
    for i in range(n):
        val = int(round(base * (ratio ** i)))
        if val < 1:
            val = 1
        if val > max_val:
            return None
        if val in A:
            return None
        A.append(val)
    return A

def _local_search(A, deadline, max_iters=50000):
    """Local search with collision-seeking moves."""
    n = len(A)
    A = list(A)
    current_val = _compute_objective(A)
    best_A = list(A)
    best_val = current_val
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Randomly pick an element to modify
        idx = random.randint(0, n - 1)
        old_val = A[idx]
        
        # Try to find a new value that creates collisions
        # Strategy: try values that make A[idx]+A[j] collide with existing sums
        # or A[idx]*A[j] collide with existing products
        candidates = set()
        
        # Collision-seeking: for each other element A[j], 
        # if we want A[idx]+A[j] to equal some existing sum s,
        # then A[idx] = s - A[j]
        sums_set = set()
        prods_set = set()
        for i in range(n):
            for j in range(i, n):
                sums_set.add(A[i] + A[j])
                prods_set.add(A[i] * A[j])
        
        for j in range(n):
            if j == idx:
                continue
            # Sums: want A[idx] + A[j] in sums_set
            for s in list(sums_set)[:50]:  # limit for speed
                cand = s - A[j]
                if 1 <= cand <= 10000 and cand != old_val:
                    candidates.add(cand)
            # Products: want A[idx] * A[j] in prods_set
            for p in list(prods_set)[:50]:
                if p % A[j] == 0:
                    cand = p // A[j]
                    if 1 <= cand <= 10000 and cand != old_val:
                        candidates.add(cand)
        
        # Also try random perturbations
        for _ in range(5):
            delta = random.randint(-20, 20)
            cand = old_val + delta
            if 1 <= cand <= 10000 and cand != old_val:
                candidates.add(cand)
        
        # Try a few random values
        for _ in range(3):
            cand = random.randint(1, 10000)
            candidates.add(cand)
        
        # Remove old value from set for checking
        A_set = set(A)
        A_set.discard(old_val)
        
        best_candidate = None
        best_candidate_val = current_val
        
        for cand in candidates:
            if cand in A_set:
                continue  # must be distinct
            A[idx] = cand
            new_val = _compute_objective(A)
            if new_val < best_candidate_val:
                best_candidate_val = new_val
                best_candidate = cand
            A[idx] = old_val
        
        if best_candidate is not None:
            A[idx] = best_candidate
            current_val = best_candidate_val
            if current_val < best_val:
                best_val = current_val
                best_A = list(A)
        else:
            # No improvement, try a random restart of one element
            idx = random.randint(0, n - 1)
            A[idx] = random.randint(1, 10000)
            # Check distinctness
            A_set = set(A)
            if len(A_set) < n:
                A[idx] = old_val
                continue
            new_val = _compute_objective(A)
            if new_val < current_val:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_A = list(A)
            else:
                A[idx] = old_val
    
    return best_A, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    random.seed(42)
    
    best_A = None
    best_val = float('inf')
    
    # Phase 1: Try geometric progressions with various parameters
    for base in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        if time.time() > deadline:
            break
        for ratio in [1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
            if time.time() > deadline:
                break
            A = _geometric_init(base, ratio)
            if A is None:
                continue
            val = _compute_objective(A)
            if val < best_val:
                best_val = val
                best_A = list(A)
    
    # Phase 2: Local search from best geometric
    if best_A is not None:
        if time.time() < deadline - 5:
            best_A, best_val = _local_search(best_A, deadline - 2)
    
    # Phase 3: Try more structured initializations
    if time.time() < deadline - 10:
        # Try powers of 2 with some perturbation
        for base in [1, 2, 3]:
            A = []
            for i in range(20):
                val = base * (2 ** i)
                if val > 10000:
                    val = 10000
                if val in A:
                    val += 1
                A.append(val)
            if len(set(A)) == 20 and all(1 <= x <= 10000 for x in A):
                val = _compute_objective(A)
                if val < best_val:
                    best_val = val
                    best_A = list(A)
        
        # Local search from these too
        if time.time() < deadline - 5:
            A2, v2 = _local_search(best_A, deadline - 2)
            if v2 < best_val:
                best_val = v2
                best_A = A2
    
    # Phase 4: Multiple random restarts with local search
    while time.time() < deadline - 5:
        # Random structured init
        A = []
        base = random.randint(1, 30)
        ratio = random.uniform(1.05, 2.0)
        for i in range(20):
            val = int(round(base * (ratio ** i)))
            if val < 1:
                val = 1
            if val > 10000:
                val = 10000
            if val in A:
                val = random.randint(1, 10000)
            if val in A:
                continue
            A.append(val)
        
        while len(A) < 20:
            val = random.randint(1, 10000)
            if val not in A:
                A.append(val)
        
        A = A[:20]
        if len(set(A)) != 20:
            continue
        
        val = _compute_objective(A)
        if val < best_val:
            best_val = val
            best_A = list(A)
        
        # Quick local search
        if time.time() < deadline - 3:
            A_ls, v_ls = _local_search(A, deadline - 2, max_iters=5000)
            if v_ls < best_val:
                best_val = v_ls
                best_A = A_ls
    
    return best_A

