"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
_start_time = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - _start_time

def _score_of(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)"""
    U = sorted(set(U))
    if len(U) < 2 or max(U) == 0:
        return 0.0
    # Compute U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    # Compute U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    log_ratio = math.log(len(diffset) / len(sumset)) / math.log(2 * max(U) + 1)
    return 1.0 + log_ratio

def _score_of_fast(U):
    """Faster score computation using numpy"""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if len(U) < 2 or U[-1] == 0:
        return 0.0
    # Compute U+U
    sumset = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sumset.add(int(U[i] + U[j]))
            if i != j:
                sumset.add(int(U[j] + U[i]))  # redundant but safe
    # Actually, since addition is commutative, we just need i <= j
    sumset = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sumset.add(int(U[i] + U[j]))
    
    # Compute U-U
    diffset = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diffset.add(int(U[i] - U[j]))
    
    log_ratio = math.log(len(diffset) / len(sumset)) / math.log(2 * int(U[-1]) + 1)
    return 1.0 + log_ratio

def _gen_geometric(base, k):
    """Generate {0, base^0, base^1, ..., base^(k-1)}"""
    U = [0]
    val = 1
    for i in range(k):
        U.append(val)
        val *= base
    return U

def _gen_arithmetic(n):
    """Generate {0, 1, 2, ..., n-1}"""
    return list(range(n))

def _gen_powers(base, k):
    """Generate {0, base^0, base^1, ..., base^(k-1)}"""
    U = [0]
    val = 1
    for i in range(k):
        U.append(val)
        val *= base
    return U

def _gen_sparse_random(n, max_val, seed=42):
    """Generate a random sparse set of size n with 0 included"""
    np.random.seed(seed)
    vals = np.random.choice(range(1, max_val + 1), size=n-1, replace=False)
    U = sorted([0] + list(vals))
    return U

def _gen_mixed(base, k, extra_n, max_extra):
    """Geometric plus some random extras"""
    U = set(_gen_geometric(base, k))
    if extra_n > 0:
        np.random.seed(123)
        extras = np.random.choice(range(1, max_extra + 1), size=extra_n, replace=False)
        U.update(extras.tolist())
    return sorted(U)

def _local_search(U, iterations=100, timeout=None):
    """Try to improve U by adding/removing elements"""
    U = list(U)
    best_score = _score_of_fast(U)
    best_U = list(U)
    
    for _ in range(iterations):
        if timeout and _elapsed() > timeout:
            break
        
        # Try adding a random element
        max_val = max(U) if U else 0
        new_val = int(np.random.randint(1, max_val * 2 + 10))
        if new_val not in U:
            U_new = U + [new_val]
            s = _score_of_fast(U_new)
            if s > best_score:
                best_score = s
                best_U = U_new
                U = U_new
        
        # Try removing a random element (not 0)
        if len(U) > 1:
            idx = int(np.random.randint(1, len(U)))
            U_new = U[:idx] + U[idx+1:]
            s = _score_of_fast(U_new)
            if s > best_score:
                best_score = s
                best_U = U_new
                U = U_new
    
    return best_U, best_score

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    np.random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = _score_of_fast(best_U)
    
    # Try various geometric constructions
    for base in [2, 3, 4, 5, 6, 7, 8, 10]:
        for k in range(3, 20):
            if _elapsed() > _TIMEOUT:
                return best_U
            U = _gen_geometric(base, k)
            s = _score_of_fast(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Try arithmetic progressions
    for n in range(3, 50):
        if _elapsed() > _TIMEOUT:
            return best_U
        U = _gen_arithmetic(n)
        s = _score_of_fast(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Try mixed constructions
    for base in [2, 3, 4, 5]:
        for k in range(3, 15):
            for extra_n in [0, 5, 10, 20]:
                if _elapsed() > _TIMEOUT:
                    return best_U
                max_extra = base ** k
                U = _gen_mixed(base, k, extra_n, max_extra)
                s = _score_of_fast(U)
                if s > best_score:
                    best_score = s
                    best_U = U
    
    # Try random sparse sets
    for n in [10, 20, 50, 100, 200]:
        for max_val in [100, 500, 1000, 5000]:
            if _elapsed() > _TIMEOUT:
                return best_U
            U = _gen_sparse_random(n, max_val, seed=n * 1000 + max_val)
            s = _score_of_fast(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Local search on the best candidate
    if _elapsed() < _TIMEOUT - 10:
        best_U, best_score = _local_search(best_U, iterations=50, timeout=_TIMEOUT - 5)
    
    # Ensure 0 is in the set and all are nonnegative
    best_U = sorted(set(x for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U = [0] + best_U
    
    return best_U
# EVOLVE-BLOCK-END

