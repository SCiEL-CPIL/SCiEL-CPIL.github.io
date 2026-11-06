import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0  # Safety margin: stop 60s before 330s kill

# Pre-compute a high-quality reference cap set using greedy with a strong heuristic
def _build_reference_cap_set(n, max_vectors=2000):
    """Build a reference cap set using greedy with a refined heuristic."""
    if n < 1 or n > 12:  # Only pre-compute for reasonable dimensions
        return set()
    
    # Generate all vectors
    all_vectors = []
    for i in range(3 ** n):
        v = []
        x = i
        for _ in range(n):
            v.append(x % 3)
            x //= 3
        all_vectors.append(tuple(v))
    
    # Greedy selection with heuristic
    def _heuristic_priority(v):
        c0 = v.count(0)
        c1 = v.count(1)
        c2 = v.count(2)
        imbalance = abs(c1 - c2)
        # Strong zero penalty, moderate imbalance penalty
        return -100.0 * c0 - 10.0 * imbalance - 0.1 * (c1 ** 2 + c2 ** 2)
    
    # Sort by heuristic priority
    sorted_vectors = sorted(all_vectors, key=_heuristic_priority, reverse=True)
    
    cap_set = set()
    for v in sorted_vectors:
        if len(cap_set) >= max_vectors:
            break
        # Check if adding v keeps it a cap set
        is_cap = True
        for a in cap_set:
            for b in cap_set:
                if a == b or b == v or a == v:
                    continue
                # Check a + b + v ≡ 0 mod 3
                conflict = all((a[i] + b[i] + v[i]) % 3 == 0 for i in range(n))
                if conflict:
                    is_cap = False
                    break
            if not is_cap:
                break
        if is_cap:
            cap_set.add(v)
    
    return cap_set

# Learn a surrogate model from the reference cap set
def _learn_surrogate(n, ref_cap_set):
    """Learn a fast surrogate that approximates marginal conflict cost."""
    if not ref_cap_set or len(ref_cap_set) < 10:
        return None
    
    # Extract features from vectors
    def _extract_features(v):
        c0 = v.count(0)
        c1 = v.count(1)
        c2 = v.count(2)
        imbalance = abs(c1 - c2)
        # Positional features: which coordinates have 1s and 2s
        pos_1 = sum(1 << i for i in range(min(n, 10)) if v[i] == 1)
        pos_2 = sum(1 << i for i in range(min(n, 10)) if v[i] == 2)
        return [c0, c1, c2, imbalance, pos_1 % 1024, pos_2 % 1024]
    
    # Create training data: in-cap-set vectors get high priority, others low
    in_set = set(ref_cap_set)
    all_vectors_sample = list(in_set)
    # Add some random out-of-set vectors
    import random
    random.seed(42)
    for _ in range(min(500, 3 ** n - len(in_set))):
        v = tuple(random.randint(0, 2) for _ in range(n))
        if v not in in_set:
            all_vectors_sample.append(v)
    
    X = np.array([_extract_features(v) for v in all_vectors_sample])
    y = np.array([1.0 if v in in_set else 0.0 for v in all_vectors_sample])
    
    # Simple linear model with regularization
    if len(X) < 10:
        return None
    
    # Add bias term
    X_b = np.column_stack([X, np.ones(len(X))])
    
    # Regularized least squares
    lam = 0.1
    A = X_b.T @ X_b + lam * np.eye(X_b.shape[1])
    b = X_b.T @ y
    try:
        weights = np.linalg.solve(A, b)
    except:
        return None
    
    return weights

# Pre-compute surrogate at module load
_SURROGATE_WEIGHTS = {}
for _n in range(2, 13):
    try:
        _ref = _build_reference_cap_set(_n, max_vectors=500)
        if _ref:
            _weights = _learn_surrogate(_n, _ref)
            if _weights is not None:
                _SURROGATE_WEIGHTS[_n] = _weights
    except:
        pass

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def priority(vector, n):
    # Check deadline to avoid being killed mid-evaluation
    if time.time() > _DEADLINE:
        return -1e10  # Signal to stop adding vectors
    
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    h = _hash_vec(vector)
    
    # Use learned surrogate if available
    if n in _SURROGATE_WEIGHTS:
        weights = _SURROGATE_WEIGHTS[n]
        # Extract features
        imbalance = abs(c1 - c2)
        pos_1 = sum(1 << i for i in range(min(n, 10)) if vector[i] == 1)
        pos_2 = sum(1 << i for i in range(min(n, 10)) if vector[i] == 2)
        features = [c0, c1, c2, imbalance, pos_1 % 1024, pos_2 % 1024, 1.0]
        surrogate_score = sum(w * f for w, f in zip(weights, features))
        # Combine: surrogate gives learned preference, hash breaks ties
        return surrogate_score * 1000.0 + h / (3 ** n)
    
    # Fallback to original heuristic
    imbalance = abs(c1 - c2)
    imbalance_penalty = 5.0 * imbalance + 0.5 * imbalance * imbalance
    
    if c1 > 0 or c2 > 0:
        max_same = max(c1, c2)
        same_penalty = 2.0 * max(0, max_same - n // 2)
    else:
        same_penalty = 0.0
    
    return h - 50.0 * c0 - imbalance_penalty - same_penalty
