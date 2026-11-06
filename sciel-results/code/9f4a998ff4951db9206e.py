import time
import zlib
import math

_START = time.time()
_DEADLINE = _START + 280.0

# Global state for the learned model
_model_trained = False
_model_w = None  # weights for the linear model
_model_bias = 0.0
_model_feature_scale = 1.0


def _elapsed():
    return time.time() - _START


def _deadline_reached():
    return _elapsed() > _DEADLINE


def _extract_features(vector, n):
    """Extract features from a vector for the learned model.
    Features: c0/n, c1/n, c2/n, (c0/n)^2, (c1/n)^2, (c2/n)^2,
              c0*c1/n^2, c0*c2/n^2, c1*c2/n^2, sqrt(c1/n), c1^2/n^2
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    if n == 0:
        return [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    f0 = c0 / n
    f1 = c1 / n
    f2 = c2 / n
    
    # Polynomial and interaction features, normalized to [0,1] range
    feats = [
        1.0,           # bias
        f0,            # linear
        f1,
        f2,
        f0 * f0,       # quadratic
        f1 * f1,
        f2 * f2,
        f0 * f1,       # interactions
        f0 * f2,
        f1 * f2,
        math.sqrt(f1) if f1 > 0 else 0.0,  # sqrt for finer resolution near 0
    ]
    return feats


def _baseline_priority(vector, n):
    """Simple baseline: fewer 1s is better, with hash tiebreak."""
    c1 = vector.count(1)
    # Exponential decay on count of 1s
    if n > 0:
        one_score = math.exp(-0.8 * c1 / n)
    else:
        one_score = 1.0
    
    # Balance between 0 and 2
    c0 = vector.count(0)
    c2 = vector.count(2)
    if c0 + c2 > 0:
        balance = 2.0 * min(c0, c2) / (c0 + c2)
    else:
        balance = 0.0
    
    # Hash for deterministic tie-breaking
    data = bytes(vector)
    crc = zlib.crc32(data)
    hash_score = float(crc) / 0xFFFFFFFF
    
    return 0.75 * one_score + 0.15 * balance + 0.10 * hash_score


def _learned_priority(vector, n):
    """Priority from the learned linear model."""
    if _model_w is None:
        return _baseline_priority(vector, n)
    
    feats = _extract_features(vector, n)
    score = _model_bias
    for i, w in enumerate(_model_w):
        score += w * feats[i]
    return score


def _is_cap_set_member_ok(vec, current_set, n):
    """Check if adding vec to current_set keeps it a cap set.
    A cap set has no 3 distinct elements a,b,c with a+b+c ≡ 0 mod 3.
    Equivalently, for any a,b in the set, the third element c = -a-b mod 3
    must not also be in the set.
    """
    for a in current_set:
        for b in current_set:
            if a == b:
                continue
            # Check if a, b, vec form a 3-AP (a+b+vec ≡ 0 mod 3)
            # Or a, vec, b form one, etc.
            # Actually we need: no three DISTINCT elements sum to 0 mod 3.
            # So check: for each pair (x,y) in {a, b, vec}, the third z
            # must not satisfy x+y+z ≡ 0 mod 3.
            # More efficiently: for each pair (a, b) in current_set,
            # check if (-a-b) mod 3 == vec.
            # And for each a in current_set, check if (-a-vec) mod 3 is in current_set.
            # And for each b in current_set, check if (-b-vec) mod 3 is in current_set.
            # Actually the cleanest: for each a in current_set, the "partner"
            # that would complete a 3-AP with a and vec is (-a-vec) mod 3.
            # If that partner is in current_set, we have a 3-AP.
            # Also need to check pairs within current_set that involve vec differently.
            # 
            # Let me think again. A 3-AP in F_3^n is a set {x, x+d, x+2d} for d≠0.
            # Equivalently, {a, b, c} with a+b+c = 0 mod 3 and a,b,c distinct.
            # 
            # When adding vec to S, we need to check:
            # 1. For any a in S: is there b in S such that a + b + vec = 0? 
            #    i.e., b = (-a - vec) mod 3. If b in S and b != a and b != vec, bad.
            # 2. For any a, b in S: is a + b + vec = 0? Same as above.
            # 3. For any a in S: is a + vec + vec = 0? i.e., a = (-2*vec) mod 3 = (vec) mod 3?
            #    That would mean a = vec, but vec not in S yet, so a != vec. 
            #    Wait: a + 2*vec = 0 mod 3 means a = -2*vec = vec (mod 3) since -2 ≡ 1 mod 3.
            #    So if vec is in S, but it's not yet. So this case doesn't apply.
            # 
            # Actually the simplest check: for each a in S, compute c = (-a - vec) mod 3.
            # If c is in S and c != a, then {a, c, vec} is a 3-AP (since a+c+vec=0).
            # Also need: for each pair a,b in S, check if a+b+vec=0, which is the same 
            # as b = (-a-vec) mod 3 being in S.
            # 
            # So: for each a in S, let c = tuple((-a[i] - vec[i]) % 3 for i in range(n)).
            # If c in S and c != a, return False.
            # Also check: is vec in S? No, we're adding it.
            # 
            # Wait, I also need to check if there's a 3-AP among {a, b, vec} where 
            # the sum is 0 but not in the form a + b + vec. 
            # Actually a+b+c=0 is symmetric, so checking a+b+vec=0 for all a,b in S 
            # covers all cases where vec is one of the three.
    
    for a in current_set:
        c = tuple((-a[i] - vec[i]) % 3 for i in range(n))
        if c in current_set and c != a:
            return False
    
    # Also check: could vec, vec, a form a 3-AP? No, they need to be distinct.
    # Could a, vec, vec? No, distinct.
    # What about a, b where a+b = 0 and vec = 0? 
    # a + b + 0 = 0 means a+b=0, i.e., b = -a. 
    # This is already covered: for a in S, c = (-a - 0) = -a. If -a in S and -a != a, 
    # then {a, -a, 0} is a 3-AP. But this only matters if vec = (0,0,...,0).
    # The check above handles this: c = (-a[i] - 0) = (-a[i]) % 3.
    
    return True


def _greedy_construct(n, priority_fn, max_time=60.0):
    """Greedy cap set construction. Returns the set of vectors added."""
    start = time.time()
    current_set = set()
    
    # Generate all vectors in order of priority
    # For efficiency, generate vectors sorted by priority
    # For small n, enumerate all 3^n vectors
    if n <= 10:
        total = 3 ** n
        vectors = []
        for i in range(total):
            vec = []
            x = i
            for _ in range(n):
                vec.append(x % 3)
                x //= 3
            vectors.append(tuple(vec))
        
        # Sort by priority (descending)
        # To save time, only sort if we have time
        if time.time() - start < 5:
            vectors.sort(key=lambda v: priority_fn(v, n), reverse=True)
        
        for vec in vectors:
            if time.time() - start > max_time:
                break
            if _is_cap_set_member_ok(vec, current_set, n):
                current_set.add(vec)
    else:
        # For larger n, use a smarter enumeration
        # Generate vectors with few 1s first
        # This is a heuristic for larger n
        import itertools
        # We can't enumerate 3^n for large n, so use a different strategy
        # For now, just return empty or a small set
        # Actually, for the FunSearch contract, n is likely small (<=10 or so)
        # Let's handle it gracefully
        pass
    
    return current_set


def _train_model(n, training_data, max_time=30.0):
    """Train a simple linear model with gradient descent.
    training_data: list of (features, label) where label=1 if added, 0 if rejected.
    """
    global _model_w, _model_bias, _model_trained
    
    if not training_data:
        _model_trained = True
        _model_w = [0.0] * 11
        _model_bias = 0.5
        return
    
    # Separate features and labels
    X = [d[0] for d in training_data]
    y = [d[1] for d in training_data]
    
    n_samples = len(X)
    n_features = len(X[0])
    
    # Initialize weights
    w = [0.0] * n_features
    bias = 0.5
    
    # Simple gradient descent with regularization
    lr = 0.1
    reg = 0.01
    max_iter = min(2000, max(100, int(max_time * 100)))
    
    start = time.time()
    for it in range(max_iter):
        if time.time() - start > max_time:
            break
        
        # Compute gradients
        dw = [0.0] * n_features
        db = 0.0
        
        for i in range(n_samples):
            z = bias
            for j in range(n_features):
                z += w[j] * X[i][j]
            # Sigmoid
            if z > 50:
                sig = 1.0
            elif z < -50:
                sig = 0.0
            else:
                sig = 1.0 / (1.0 + math.exp(-z))
            
            err = sig - y[i]
            db += err
            for j in range(n_features):
                dw[j] += err * X[i][j]
        
        # Update
        scale = 1.0 / n_samples
        for j in range(n_features):
            w[j] -= lr * (dw[j] * scale + reg * w[j])
        bias -= lr * db * scale
    
    _model_w = w
    _model_bias = bias
    _model_trained = True


def priority(vector, n):
    """
    Learned priority for cap set construction.
    
    Strategy:
    1. First call: run baseline greedy to collect training data, train model.
    2. Subsequent calls: use learned model for priority.
    
    The learned model captures higher-order interactions in the count distribution
    that better predict which vectors should be prioritized.
    """
    global _model_trained, _model_w
    
    # Deadline guard
    if _deadline_reached():
        return 0.0
    
    if not _model_trained:
        # First call: we need to train the model.
        # Run a quick baseline greedy to collect training data.
        # But we can't run the full greedy here because priority is called 
        # during the greedy process by the evaluator.
        # 
        # Instead, we use a two-phase approach:
        # Phase 1: Use baseline priority, but track which vectors are added/rejected.
        # Phase 2: After enough data, train and switch to learned priority.
        #
        # Since we don't have access to the evaluator's internal state,
        # we'll use a different strategy: 
        # - Pre-compute a good ordering using baseline
        # - The first few calls will use baseline, and we'll try to 
        #   infer the set state from the vectors we've seen.
        
        # Actually, the cleanest approach: 
        # On the very first call, do a quick offline training using 
        # synthetic data or a fast greedy simulation.
        
        # Let's do a fast greedy simulation to get training data
        sim_start = time.time()
        sim_set = set()
        training_data = []
        
        # Generate a sample of vectors (not all, for speed)
        if n <= 8:
            total = 3 ** n
            # Sample up to 5000 vectors
            import random
            random.seed(42)
            if total <= 5000:
                sample_indices = list(range(total))
            else:
                sample_indices = random.sample(range(total), 5000)
            
            # Get baseline priorities
            vec_priorities = []
            for idx in sample_indices:
                vec = []
                x = idx
                for _ in range(n):
                    vec.append(x % 3)
                    x //= 3
                vec = tuple(vec)
                p = _baseline_priority(vec, n)
                vec_priorities.append((p, vec))
            
            # Sort by priority
            vec_priorities.sort(key=lambda x: x[0], reverse=True)
            
            # Greedy simulation
            for p, vec in vec_priorities:
                if time.time() - sim_start > 80:
                    break
                ok = _is_cap_set_member_ok(vec, sim_set, n)
                if ok:
                    sim_set.add(vec)
                    training_data.append((_extract_features(vec, n), 1))
                else:
                    training_data.append((_extract_features(vec, n), 0))
        
        # Train the model
        _train_model(n, training_data, max_time=20.0)
        _model_trained = True
    
    # Use the learned model
    return _learned_priority(vector, n)
