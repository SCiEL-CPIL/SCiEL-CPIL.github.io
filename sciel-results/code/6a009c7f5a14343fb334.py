import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 80.0  # seconds, safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _evaluate(a):
    """Return (|A+A|, |A*A|, max)."""
    a = np.asarray(a, dtype=np.int64)
    n = len(a)
    s = set()
    for i in range(n):
        for j in range(i, n):
            s.add(a[i] + a[j])
    p = set()
    for i in range(n):
        for j in range(i, n):
            p.add(a[i] * a[j])
    return len(s), len(p), max(len(s), len(p))

def _evaluate_np(a):
    """Faster evaluate using numpy for sums."""
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    # Sumset
    s = set()
    for i in range(n):
        vals = a[i] + a[i:]
        s.update(vals.tolist())
    # Productset
    p = set()
    for i in range(n):
        vals = a[i] * a[i:]
        p.update(vals.tolist())
    return len(s), len(p), max(len(s), len(p))

def _is_valid(a, n=20, maximum=10000):
    if len(a) != n:
        return False
    if len(set(a)) != n:
        return False
    if min(a) < 1:
        return False
    if max(a) > maximum:
        return False
    return True

def _generate_geometric(n=20, max_val=10000, seed=0):
    """Generate a set with geometric structure to minimize product set."""
    rng = np.random.RandomState(seed)
    # Start with small base and multiply
    # Use a base with many small prime factors
    best = None
    best_score = float('inf')
    
    for trial in range(50):
        # Pick a base number with many divisors
        # Try bases that are products of small primes
        base = 1
        factors = [2, 3, 5, 7, 11]
        # Randomly choose exponents
        for f in factors:
            e = rng.randint(0, 4)
            base *= (f ** e)
        if base < 1 or base > max_val:
            continue
        
        # Generate multiples: base * k for k in some set
        # To keep product set small, use k values that are also from a small multiplicative group
        # Try: a_i = base * 2^i for i in range
        # Or a_i = base * (small set of multipliers)
        
        # Strategy 1: geometric progression
        ratio = rng.choice([2, 3, 4, 5, 6])
        vals = []
        v = base
        while len(vals) < n and v <= max_val:
            vals.append(v)
            v *= ratio
        if len(vals) < n:
            # Fill in with nearby values
            for k in range(1, max_val):
                if len(vals) >= n:
                    break
                if k not in vals and k <= max_val:
                    vals.append(k)
        if len(vals) == n and len(set(vals)) == n:
            s, p, mx = _evaluate_np(vals)
            if mx < best_score:
                best_score = mx
                best = list(vals)
        
        # Strategy 2: powers of 2 scaled
        scale = rng.randint(1, max(1, max_val // (2**n)))
        if scale >= 1:
            vals = [scale * (2**i) for i in range(n)]
            if all(v <= max_val for v in vals) and len(set(vals)) == n:
                s, p, mx = _evaluate_np(vals)
                if mx < best_score:
                    best_score = mx
                    best = list(vals)
        
        # Strategy 3: mixed - some geometric, some clustered
        # Use values close to each other to minimize sumset
        center = rng.randint(1, max_val // 2)
        spread = rng.randint(1, 50)
        vals = []
        used = set()
        for _ in range(n):
            v = center + rng.randint(-spread, spread)
            if 1 <= v <= max_val and v not in used:
                used.add(v)
                vals.append(v)
        if len(vals) == n:
            s, p, mx = _evaluate_np(vals)
            if mx < best_score:
                best_score = mx
                best = list(vals)
    
    return best, best_score

def _local_search(a, n=20, max_val=10000, time_budget=50.0):
    """Local search to improve the set."""
    a = list(a)
    s, p, best_score = _evaluate_np(a)
    best_a = list(a)
    deadline = time.time() + time_budget
    
    # Hill climbing with random restarts
    while time.time() < deadline:
        improved = False
        # Try swapping or modifying elements
        for i in range(n):
            if time.time() > deadline:
                break
            # Try replacing a[i] with nearby values
            current = a[i]
            # Try small perturbations
            candidates = []
            for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10]:
                nv = current + delta
                if 1 <= nv <= max_val and nv not in a[:i] and nv not in a[i+1:]:
                    candidates.append(nv)
            # Also try some random values
            for _ in range(10):
                nv = int(np.random.randint(1, max_val + 1))
                if nv not in a[:i] and nv not in a[i+1:]:
                    candidates.append(nv)
            
            for nv in candidates:
                a[i] = nv
                _, _, sc = _evaluate_np(a)
                if sc < best_score:
                    best_score = sc
                    best_a = list(a)
                    improved = True
                    break
                a[i] = current
        
        if not improved:
            # Random restart from best
            a = list(best_a)
            # Perturb multiple elements
            for _ in range(3):
                idx = int(np.random.randint(0, n))
                a[idx] = int(np.random.randint(1, max_val + 1))
            # Make valid
            while len(set(a)) < n or min(a) < 1 or max(a) > max_val:
                idx = int(np.random.randint(0, n))
                a[idx] = int(np.random.randint(1, max_val + 1))
            s, p, sc = _evaluate_np(a)
            if sc < best_score:
                best_score = sc
                best_a = list(a)
    
    return best_a, best_score

def _structured_candidates(n=20, max_val=10000):
    """Generate several structured candidate sets."""
    candidates = []
    
    # Geometric progressions
    for ratio in [2, 3, 4, 5]:
        for start in [1, 2, 3, 4, 5, 10, 20]:
            vals = [start * (ratio ** i) for i in range(n)]
            if all(v <= max_val for v in vals) and len(set(vals)) == n:
                candidates.append(vals)
    
    # Near-geometric with small perturbations
    base = 2
    ratio = 2
    vals = [base * (ratio ** i) for i in range(n)]
    if all(v <= max_val for v in vals):
        candidates.append(vals)
    
    # Clustered (arithmetic progression)
    for start in [1, 10, 50, 100]:
        for diff in [1, 2, 3, 5, 10]:
            vals = [start + i * diff for i in range(n)]
            if all(1 <= v <= max_val for v in vals) and len(set(vals)) == n:
                candidates.append(vals)
    
    # Powers of 2
    vals = [2**i for i in range(n)]
    if all(v <= max_val for v in vals):
        candidates.append(vals)
    
    # Mixed: powers of 2 times a constant
    for c in [1, 2, 3, 4, 5]:
        vals = [c * (2**i) for i in range(n)]
        if all(v <= max_val for v in vals) and len(set(vals)) == n:
            candidates.append(vals)
    
    return candidates

def construct():
    # Deadline guard
    if time.time() - _START_TIME > 55.0:
        return list(range(1, 21))
    
    n = 20
    max_val = 10000
    
    # Phase 1: Generate structured candidates
    best_a = list(range(1, 21))
    best_score = _evaluate_np(best_a)[2]
    
    # Try structured candidates
    candidates = _structured_candidates(n, max_val)
    for cand in candidates:
        if time.time() - _START_TIME > 55.0:
            break
        s, p, sc = _evaluate_np(cand)
        if sc < best_score:
            best_score = sc
            best_a = list(cand)
    
    # Phase 2: Geometric generation
    geo_a, geo_score = _generate_geometric(n, max_val, seed=42)
    if geo_a is not None and geo_score < best_score:
        best_score = geo_score
        best_a = list(geo_a)
    
    # Phase 3: Local search
    if time.time() - _START_TIME < 50.0:
        remaining = min(35.0, 55.0 - (time.time() - _START_TIME))
        best_a, best_score = _local_search(best_a, n, max_val, remaining)
    
    # Validate
    if not _is_valid(best_a, n, max_val):
        best_a = list(range(1, 21))
    
    return best_a
