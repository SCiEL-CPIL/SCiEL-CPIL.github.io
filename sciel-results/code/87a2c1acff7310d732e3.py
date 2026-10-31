"""Optimized construction for sumset/difference-set exponent task."""
import time
import math

_DEADLINE = 150.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _score_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return -1.0
    U = list(set(U))
    n = len(U)
    if n < 2:
        return -1.0
    mx = max(U)
    if mx == 0:
        return -1.0
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    num = len(diffset)
    den = len(sumset)
    if den <= 0 or num <= 0:
        return -1.0
    ratio = num / den
    if ratio <= 0:
        return -1.0
    denom_log = math.log(2 * mx + 1)
    if denom_log <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / denom_log

def _build_binary(k):
    """Build U = {0, 1, 2, 4, ..., 2^(k-1)} (k powers of 2 including 2^0=1)."""
    return [0] + [1 << i for i in range(k)]

def _build_geometric(base, k):
    """Build U = {0, 1, base, base^2, ..., base^(k-1)}."""
    result = [0]
    val = 1
    for _ in range(k):
        result.append(val)
        val *= base
    return result

def _build_mixed(base, k):
    """Build U = {0} union {base^i} union {base^i * offset} for small offsets."""
    # Base geometric plus some shifted copies to enrich difference set
    base_set = _build_geometric(base, k)
    # Add a second "layer" shifted by a large multiple
    shift = base ** (k - 1) * 2
    result = set(base_set)
    for i in range(k):
        result.add(shift + base ** i)
    return sorted(result)

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize score."""
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Safety: if we've already used too much time, return immediately
    if _remaining() < 5.0:
        return best_U
    
    # Strategy 1: Binary (powers of 2) sets of varying sizes
    # These are fast to score because the structure is regular
    for k in range(2, 60):
        if _remaining() < 10.0:
            break
        U = _build_binary(k)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Strategy 2: Geometric sets with base 3
    for k in range(2, 30):
        if _remaining() < 10.0:
            break
        U = _build_geometric(3, k)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Strategy 3: Geometric sets with base 4
    for k in range(2, 25):
        if _remaining() < 10.0:
            break
        U = _build_geometric(4, k)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Strategy 4: Mixed sets (geometric + shifted layer) for small k
    for base in [2, 3]:
        for k in range(2, 15):
            if _remaining() < 10.0:
                break
            U = _build_mixed(base, k)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Strategy 5: Try some structured sets with gaps
    # U = {0} union {i * gap : i in range(1, k)}
    for gap in [2, 3, 5]:
        for k in range(2, 50):
            if _remaining() < 5.0:
                break
            U = [0] + [gap * i for i in range(1, k)]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Final safety check
    if _remaining() < 3.0:
        return best_U
    
    return best_U
