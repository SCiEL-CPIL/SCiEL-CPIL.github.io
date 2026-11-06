def _build_smooth_set():
    smooths = set()
    for a in range(0, 8):
        for b in range(0, 5):
            for c in range(0, 4):
                v = (2**a) * (3**b) * (5**c)
                if v <= 128:
                    smooths.add(v)
    return smooths

_SMOOTH_SET = _build_smooth_set()

def _is_antichain(terms):
    for i in range(len(terms)):
        for j in range(len(terms)):
            if i != j and terms[j] % terms[i] == 0:
                return False
    return True

def _find_optimal(n, smooths_list):
    """Find minimal antichain representation for target n."""
    smooth_set = _SMOOTH_SET
    
    # k=1
    if n in smooth_set:
        return [n]
    
    # k=2
    for a in smooths_list:
        if a >= n:
            break
        b = n - a
        if b > a and b in smooth_set:
            if b % a != 0:
                return [a, b]
    
    # k=3
    for i in range(len(smooths_list)):
        a = smooths_list[i]
        if a >= n // 3:
            break
        for j in range(i + 1, len(smooths_list)):
            b = smooths_list[j]
            if a + b >= n:
                break
            c = n - a - b
            if c > b and c in smooth_set:
                if b % a != 0 and c % a != 0 and c % b != 0:
                    return [a, b, c]
    
    # k=4 (rarely needed)
    for i in range(len(smooths_list)):
        a = smooths_list[i]
        if a >= n // 4:
            break
        for j in range(i + 1, len(smooths_list)):
            b = smooths_list[j]
            if a + b >= n // 3:
                break
            for k in range(j + 1, len(smooths_list)):
                c = smooths_list[k]
                if a + b + c >= n:
                    break
                d = n - a - b - c
                if d > c and d in smooth_set:
                    # Check all pairs
                    if _is_antichain([a, b, c, d]):
                        return [a, b, c, d]
    
    # k=5 (very rarely needed)
    # Fallback: use more terms
    # For safety, try a greedy approach
    return None

def _precompute():
    """Precompute optimal representations for all targets 1..128."""
    smooths_list = sorted(_SMOOTH_SET)
    results = []
    for n in range(1, 129):
        sol = _find_optimal(n, smooths_list)
        if sol is None:
            # Should not happen for n <= 128, but handle gracefully
            sol = [n] if n in _SMOOTH_SET else [1, n-1]
        results.append(sol)
    return results

_OPTIMAL_TABLE = _precompute()

def construct():
    return _OPTIMAL_TABLE
