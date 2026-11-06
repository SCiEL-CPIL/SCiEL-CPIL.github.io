import time
import random
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def _can_add(vec, vec_set, n):
    """Check if vec can be added to vec_set while maintaining cap set property."""
    for a in vec_set:
        for b in vec_set:
            if a == b:
                continue
            ok = True
            for k in range(n):
                if (a[k] + b[k] + vec[k]) % 3 != 0:
                    ok = False
                    break
            if ok:
                return False
    for a in vec_set:
        ok = True
        for k in range(n):
            if (2 * a[k] + vec[k]) % 3 != 0:
                ok = False
                break
        if ok:
            return False
    return True

def _greedy_cap_set(n, order):
    """Build cap set by greedily adding vectors in given order."""
    cap_set = []
    vec_set = set()
    for vec in order:
        if _can_add(vec, vec_set, n):
            cap_set.append(vec)
            vec_set.add(vec)
    return cap_set

def _hill_climb(cap_set, n, max_iter=20000):
    """Improve cap set via local search: try replacing vectors."""
    vec_set = set(cap_set)
    best_cap_set = list(cap_set)
    best_size = len(cap_set)
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        if len(cap_set) < 1:
            break
        
        idx = random.randint(0, len(cap_set) - 1)
        removed = cap_set[idx]
        new_vec_set = vec_set - {removed}
        
        # Try random vectors to replace
        found = False
        for _ in range(200):
            vec = tuple(random.randint(0, 2) for _ in range(n))
            if vec == removed or vec in new_vec_set:
                continue
            if _can_add(vec, new_vec_set, n):
                cap_set[idx] = vec
                new_vec_set.add(vec)
                found = True
                if len(new_vec_set) > best_size:
                    best_size = len(new_vec_set)
                    best_cap_set = list(new_vec_set)
                break
        
        if not found:
            vec_set.add(removed)
    
    return best_cap_set

def _compute_repulsive_potential(vector, reference_cap_set, n, bandwidth=0.8):
    """
    Compute repulsive potential from reference cap set members.
    Uses a smooth Gaussian-like kernel over the group structure.
    Lower potential = farther from existing cap set members = better candidate.
    """
    potential = 0.0
    for ref in reference_cap_set:
        # Compute distance in Z_3^n using Hamming-like metric
        # Two vectors conflict if they form a triple with any existing vector
        dist = 0
        for k in range(n):
            diff = (vector[k] - ref[k]) % 3
            # In Z_3, differences are 0, 1, or 2
            # Weight by how "close" they are in the group structure
            if diff == 0:
                dist += 1.0
            elif diff == 1 or diff == 2:
                dist += 0.5  # Adjacent in the cycle
        
        # Repulsive kernel: exp(-dist / bandwidth)
        # Higher distance = lower repulsion
        repulsion = math.exp(-dist / bandwidth)
        potential += repulsion
    
    return potential

def _build_reference_cap_set(n):
    """Build a reference cap set for computing repulsive potentials."""
    # Start with a greedy construction using a good initial ordering
    all_vectors = [tuple(random.randint(0, 2) for _ in range(n)) for _ in range(min(1000, 3**n))]
    
    # Sort by a heuristic that favors diverse vectors
    def heuristic_score(v):
        c0 = sum(1 for x in v if x == 0)
        c1 = sum(1 for x in v if x == 1)
        c2 = sum(1 for x in v if x == 2)
        return -abs(c1 - c2) + min(c1, c2) - 0.5 * c0
    
    all_vectors.sort(key=heuristic_score, reverse=True)
    
    cap_set = []
    vec_set = set()
    for vec in all_vectors:
        if _can_add(vec, vec_set, n):
            cap_set.append(vec)
            vec_set.add(vec)
    
    # Hill-climb to improve
    if time.time() < _DEADLINE - 10.0:
        cap_set = _hill_climb(cap_set, n, max_iter=5000)
    
    return cap_set

def priority(vector, n):
    # Ensure deterministic, distinct scores for all vectors
    # Use repulsive potential from a reference cap set as the main signal
    
    c0 = sum(1 for v in vector if v == 0)
    c1 = sum(1 for v in vector if v == 1)
    c2 = sum(1 for v in vector if v == 2)
    
    # Build or reuse a reference cap set
    if not hasattr(priority, '_ref_cap_set') or priority._ref_n != n:
        if time.time() > _DEADLINE - 20.0:
            # Not enough time, fall back to heuristic
            priority._ref_cap_set = []
        else:
            priority._ref_cap_set = _build_reference_cap_set(n)
        priority._ref_n = n
    
    # Compute repulsive potential (lower is better = farther from existing cap set)
    if priority._ref_cap_set:
        # Use multiple bandwidths for robustness
        pot_05 = _compute_repulsive_potential(vector, priority._ref_cap_set, n, bandwidth=0.5)
        pot_08 = _compute_repulsive_potential(vector, priority._ref_cap_set, n, bandwidth=0.8)
        pot_12 = _compute_repulsive_potential(vector, priority._ref_cap_set, n, bandwidth=1.2)
        
        # Smooth minimum via log-sum-exp (soft-min)
        # We want low potential, so use -potential for maximization
        vals = [-pot_05, -pot_08, -pot_12]
        max_val = max(vals)
        log_sum_exp = max_val + math.log(sum(math.exp(v - max_val) for v in vals))
        soft_min_potential = -log_sum_exp
    else:
        soft_min_potential = 0.0
    
    # Hash for uniqueness
    h = _hash_vec(vector)
    hash_component = (h / 4294967296.0) * 100.0
    
    # Structural features
    imbalance = abs(c1 - c2)
    structural = -3.0 * c0 - 1.5 * imbalance + 0.8 * min(c1, c2)
    
    # Combine: repulsive potential dominates, then structural, then hash
    result = soft_min_potential * 10.0 + structural + hash_component
    
    return result
