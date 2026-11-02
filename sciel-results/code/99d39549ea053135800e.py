import math
import time
import random
from itertools import combinations
import numpy as np
from collections import Counter

def score(U):
    """Compute the score for a finite set U containing 0."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0:
        return 0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
        
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def _fast_score(U):
    """Optimized score calculation using numpy for vectorized operations."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(U_arr.max())
    if max_u == 0:
        return 1.0
    
    # Vectorized sumset and diffset computation
    sumset = np.unique(U_arr[:, None] + U_arr[None, :])
    diffset = np.unique(U_arr[:, None] - U_arr[None, :])
    
    n_sums = len(sumset)
    n_diffs = len(diffset)
    
    if n_sums == 0:
        return 0.0
        
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _score_symmetric(S, M):
    """Score the symmetric set U = {0} ∪ {x, M-x for x in S} without building full arrays.
    
    Key insight: For U = {0} ∪ {x_i, M-x_i}, the sumset concentrates around M and 2M
    while the difference set spreads across [-M, M].
    """
    if not S or M <= 0:
        return 0.0
    
    # Build the set U
    U = [0]
    for x in S:
        if x != M - x:
            U.append(x)
            U.append(M - x)
        else:
            U.append(x)
    U = sorted(set(U))
    
    if len(U) < 2:
        return 0.0
    
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Compute sumset and diffset sizes efficiently
    # Sumset: all sums a+b for a,b in U
    # Diffset: all differences a-b for a,b in U
    
    # For symmetric sets, we can be smarter:
    # Sumset elements: 0+0=0, 0+x=x, 0+(M-x)=M-x, x+y, x+(M-y), (M-x)+y, (M-x)+(M-y)
    # The sums x+(M-y) = M + (x-y) concentrate around M
    # The sums (M-x)+(M-y) = 2M - (x+y) concentrate around 2M
    
    n = len(U)
    # For n up to ~30, direct computation is fine
    if n <= 40:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        n_sums = len(sums)
        n_diffs = len(diffs)
    else:
        # Use numpy for larger sets
        U_arr = np.array(U, dtype=np.int64)
        sumset = np.unique(U_arr[:, None] + U_arr[None, :])
        diffset = np.unique(U_arr[:, None] - U_arr[None, :])
        n_sums = len(sumset)
        n_diffs = len(diffset)
    
    if n_sums == 0:
        return 0.0
        
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _search_symmetric_sets(deadline=None):
    """Search over symmetric core structures U = {0} ∪ {x, M-x | x ∈ S}."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    # Try various moduli M and small subsets S
    # The key is that S should be a "Sidon-like" set to maximize differences
    # while M-x creates concentration in sums
    
    for M in [10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 500, 700, 1000]:
        if deadline and time.time() > deadline:
            break
        
        # Try small S of various sizes
        for s_size in range(1, min(8, M // 3 + 1)):
            if deadline and time.time() > deadline:
                break
            
            # Generate candidate S values (avoiding M/2 to prevent duplicates)
            valid_range = [x for x in range(1, M) if x != M - x]
            
            if len(valid_range) < s_size:
                continue
            
            # Try structured S: powers, geometric
            # Powers of 2
            if s_size <= 6:
                S_pow = [2**i for i in range(s_size) if 2**i < M and 2**i != M - 2**i]
                if len(S_pow) == s_size:
                    s_val = _score_symmetric(S_pow, M)
                    if s_val > best_score:
                        best_score = s_val
                        best_U = _symmetric_to_U(S_pow, M)
            
            # Geometric with base 3
            if s_size <= 5:
                S_geom = [3**i for i in range(s_size) if 3**i < M and 3**i != M - 3**i]
                if len(S_geom) == s_size:
                    s_val = _score_symmetric(S_geom, M)
                    if s_val > best_score:
                        best_score = s_val
                        best_U = _symmetric_to_U(S_geom, M)
            
            # Random S with local search
            for trial in range(20):
                if deadline and time.time() > deadline:
                    break
                
                # Random S
                S = random.sample(valid_range, min(s_size, len(valid_range)))
                s_val = _score_symmetric(S, M)
                
                if s_val > best_score:
                    best_score = s_val
                    best_U = _symmetric_to_U(S, M)
                
                # Local search on S: try replacing elements
                for _ in range(50):
                    if deadline and time.time() > deadline:
                        break
                    improved = False
                    idx = random.randint(0, len(S) - 1)
                    old_val = S[idx]
                    new_val = random.choice(valid_range)
                    if new_val == old_val:
                        continue
                    S_new = S[:]
                    S_new[idx] = new_val
                    # Check no duplicates
                    if len(set(S_new)) != len(S_new):
                        continue
                    s_new = _score_symmetric(S_new, M)
                    if s_new > best_score + 1e-9:
                        best_score = s_new
                        best_U = _symmetric_to_U(S_new, M)
                        S = S_new
                        improved = True
                    elif s_new > s_val:
                        # Accept if better than current (not necessarily global best)
                        S = S_new
                        s_val = s_new
                        improved = True
                    if not improved and random.random() < 0.1:
                        # Occasional restart
                        S = random.sample(valid_range, min(s_size, len(valid_range)))
                        s_val = _score_symmetric(S, M)
    
    return best_U, best_score

def _symmetric_to_U(S, M):
    """Convert symmetric representation (S, M) to the actual set U."""
    U = [0]
    for x in S:
        if x != M - x:
            U.append(x)
            U.append(M - x)
        else:
            U.append(x)
    return sorted(set(U))

def _score_set(U):
    """Score a set U containing 0."""
    U_set = sorted(set(U))
    if 0 not in U_set:
        return 0.0
    return _fast_score(U_set)

def _search_small_sets(max_n=6, max_val=80, deadline=None):
    """Exhaustive search for small sets U with numpy-accelerated scoring."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            U.sort()
            
            s = _fast_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _construct_structured_sets(deadline=None):
    """Construct sets with known good structures: powers, geometric, mixed."""
    candidates = []
    
    # Structure 1: Pure powers of 2
    for k in range(2, 15):
        U = [2**i for i in range(k)]
        candidates.append(U)
    
    # Structure 2: Powers of 3
    for k in range(2, 10):
        U = [3**i for i in range(k)]
        candidates.append(U)
    
    # Structure 3: Mixed bases - start with powers, add perturbations
    for base in [2, 3, 4, 5]:
        for k in range(2, 12):
            U = [base**i for i in range(k)]
            # Try adding a few extra elements
            for extra in range(1, 5):
                if deadline and time.time() > deadline:
                    break
                max_u = U[-1]
                new_val = random.randint(1, max_u + 20)
                if new_val not in U:
                    U_new = sorted(U + [new_val])
                    candidates.append(U_new)
    
    # Structure 4: Golomb rulers (known optimal for difference sets)
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
        [0, 1, 4, 10, 12, 17, 22, 29, 36],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    ]
    candidates.extend(known_gr)
    
    # Structure 5: Arithmetic progressions with gaps
    for start in range(0, 5):
        for step in range(2, 20):
            for length in range(3, 10):
                U = [start + i * step for i in range(length)]
                if 0 in U:
                    candidates.append(U)
                elif start > 0:
                    U_with_0 = sorted([0] + U)
                    candidates.append(U_with_0)
    
    # Structure 6: Hybrid - powers plus small AP
    for k in range(2, 8):
        powers = [2**i for i in range(k)]
        for ap_start in range(1, 10):
            for ap_step in range(1, 5):
                for ap_len in range(2, 6):
                    ap = [ap_start + i * ap_step for i in range(ap_len)]
                    U = sorted(set(powers) | set(ap))
                    if 0 in U and len(U) >= 3:
                        candidates.append(U)
    
    # Structure 7: Randomized geometric-like sets
    random.seed(42)
    for _ in range(200):
        if deadline and time.time() > deadline:
            break
        n = random.randint(3, 10)
        base = random.uniform(1.5, 3.0)
        U = [0]
        current = random.randint(1, 5)
        for i in range(n - 1):
            current = int(current * base + random.uniform(-1, 1))
            current = max(1, current)
            if current not in U:
                U.append(current)
        candidates.append(sorted(U))
    
    return candidates

def _local_search(U, max_iters=300, deadline=None):
    """Improve set U via local search with better neighborhood exploration."""
    U = sorted(set(U))
    best_score = _fast_score(U)
    best_U = U[:]
    
    for iteration in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        max_u = max(U)
        
        # Strategy 1: Try replacing each element with values in a wider range
        for i in range(len(U)):
            if i == 0 and U[0] == 0:
                continue
            old_val = U[i]
            # Try a range of new values
            for new_val in range(1, max_u + 30):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _fast_score(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        # Strategy 2: Try adding a new element
        if not improved:
            for new_val in range(1, max_u + 40):
                if new_val in U:
                    continue
                new_U = sorted(U + [new_val])
                s = _fast_score(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        # Strategy 3: Try removing an element (if size > 2)
        if not improved and len(U) > 2:
            for i in range(1, len(U)):
                new_U = U[:]
                new_U.pop(i)
                s = _fast_score(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            # Try random perturbations
            for _ in range(10):
                if deadline and time.time() > deadline:
                    break
                if random.random() < 0.5:
                    # Replace random element
                    idx = random.randint(1, len(U) - 1)
                    new_val = random.randint(1, max_u + 50)
                    new_U = U[:]
                    new_U[idx] = new_val
                else:
                    # Add random element
                    new_val = random.randint(1, max_u + 50)
                    if new_val in U:
                        continue
                    new_U = sorted(U + [new_val])
                
                if len(set(new_U)) != len(new_U):
                    continue
                s = _fast_score(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def _hill_climbing(U, max_iters=200, deadline=None):
    """Hill climbing with add/remove/replace operations."""
    current_set = sorted(set(U))
    current_score = _fast_score(current_set)
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        max_u = max(current_set) if current_set else 0
        upper_bound = max_u * 2 + 20
        
        # Try adding a random element
        x = random.randint(1, upper_bound)
        if x not in current_set:
            new_set = sorted(current_set + [x])
            s = _fast_score(new_set)
            if s > current_score + 1e-9:
                current_set = new_set
                current_score = s
                improved = True
        
        # Try removing an element (if size > 2)
        if not improved and len(current_set) > 2:
            idx = random.randint(1, len(current_set) - 1)
            new_set = current_set[:]
            new_set.pop(idx)
            s = _fast_score(new_set)
            if s > current_score + 1e-9:
                current_set = new_set
                current_score = s
                improved = True
        
        # Try replacing an element
        if not improved and len(current_set) > 1:
            idx = random.randint(1, len(current_set) - 1)
            new_val = random.randint(1, upper_bound)
            if new_val not in current_set:
                new_set = current_set[:]
                new_set[idx] = new_val
                s = _fast_score(new_set)
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    improved = True
        
        if not improved:
            # Try a few more random moves
            for _ in range(10):
                if deadline and time.time() > deadline:
                    break
                move_type = random.choice(['add', 'remove', 'replace'])
                
                if move_type == 'add':
                    x = random.randint(1, upper_bound)
                    if x not in current_set:
                        new_set = sorted(current_set + [x])
                        s = _fast_score(new_set)
                        if s > current_score + 1e-9:
                            current_set = new_set
                            current_score = s
                            improved = True
                            break
                elif move_type == 'remove' and len(current_set) > 2:
                    idx = random.randint(1, len(current_set) - 1)
                    new_set = current_set[:]
                    new_set.pop(idx)
                    s = _fast_score(new_set)
                    if s > current_score + 1e-9:
                        current_set = new_set
                        current_score = s
                        improved = True
                        break
                elif move_type == 'replace' and len(current_set) > 1:
                    idx = random.randint(1, len(current_set) - 1)
                    new_val = random.randint(1, upper_bound)
                    if new_val not in current_set:
                        new_set = current_set[:]
                        new_set[idx] = new_val
                        s = _fast_score(new_set)
                        if s > current_score + 1e-9:
                            current_set = new_set
                            current_score = s
                            improved = True
                            break
        
        if not improved:
            break
            
    return sorted(set(current_set)), current_score

def _multi_start_search(deadline=None):
    """Multi-start local search from diverse initial points."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    # Generate diverse starting points
    starts = []
    
    # Powers of 2
    for k in range(3, 10):
        starts.append([2**i for i in range(k)])
    
    # Powers of 3
    for k in range(3, 8):
        starts.append([3**i for i in range(k)])
    
    # Golomb rulers
    starts.extend([
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
    ])
    
    # Random geometric
    random.seed(123)
    for _ in range(30):
        n = random.randint(3, 8)
        base = random.uniform(1.5, 2.5)
        U = [0]
        current = random.randint(1, 5)
        for i in range(n - 1):
            current = int(current * base + random.uniform(-1, 1))
            current = max(1, current)
            if current not in U:
                U.append(current)
        starts.append(sorted(U))
    
    for U in starts:
        if deadline and time.time() > deadline:
            break
        U = sorted(set(U))
        if 0 not in U:
            continue
        s = _fast_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Local search from best start
    if best_score > 0 and deadline and time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=200, deadline=deadline)
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # 1. Symmetric core search (primary strategy)
    try:
        U_sym, s_sym = _search_symmetric_sets(deadline=deadline)
        if s_sym > best_score:
            best_score = s_sym
            best_set = U_sym
    except Exception:
        pass
    
    # 2. Exhaustive search for small sets
    try:
        U_sol, s_sol = _search_small_sets(max_n=6, max_val=60, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_set = U_sol
    except Exception:
        pass
        
    # 3. Structured candidates
    candidates = _construct_structured_sets(deadline=deadline)
    for U in candidates:
        if deadline and time.time() > deadline:
            break
        U = sorted(set(U))
        if 0 not in U or len(U) < 2:
            continue
        s = _fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # 4. Local search on best found
    if time.time() < deadline:
        best_set, best_score = _local_search(best_set, max_iters=200, deadline=deadline)
        
    # 5. Multi-start search
    if time.time() < deadline:
        U_ms, s_ms = _multi_start_search(deadline=deadline)
        if s_ms > best_score:
            best_score = s_ms
            best_set = U_ms
            
    # 6. Hill climbing for further improvement
    if time.time() < deadline:
        best_set, best_score = _hill_climbing(best_set, max_iters=150, deadline=deadline)
        
    # 7. Final local polish
    if time.time() < deadline:
        best_set, best_score = _local_search(best_set, max_iters=100, deadline=deadline)
        
    # 8. Final symmetric refinement: try to find a symmetric structure near the best set
    if time.time() < deadline - 10:
        try:
            # Extract potential symmetric structure from best_set
            max_u = max(best_set)
            for M in range(max_u, max_u + 50):
                if time.time() > deadline:
                    break
                # Find S such that U ≈ {0} ∪ {x, M-x}
                S = [x for x in best_set if x != 0 and x < M - x]
                # Check if this is close to symmetric
                U_check = _symmetric_to_U(S, M)
                if len(U_check) >= len(best_set) - 2:
                    s_sym = _score_symmetric(S, M)
                    if s_sym > best_score + 1e-6:
                        best_score = s_sym
                        best_set = U_check
        except Exception:
            pass
        
    return sorted(set(best_set))
