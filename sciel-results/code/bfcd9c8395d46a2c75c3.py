# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _generate_smooth(limit):
    """Generate all 2^a * 3^b * 5^c numbers up to limit."""
    smooth = set()
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def _is_antichain(terms):
    """Check that no term divides another in the list."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a < b:
                if b % a == 0:
                    return False
            else:
                if a % b == 0:
                    return False
    return True

def _precompute_antichains(smooth, max_size=4):
    """Precompute all valid antichains of size 1 to max_size."""
    antichains = {1: [], 2: [], 3: [], 4: []}
    L = len(smooth)
    
    # Size 1: all single terms
    for x in smooth:
        antichains[1].append((x,))
    
    # Size 2: pairs where neither divides the other
    for i in range(L):
        a = smooth[i]
        for j in range(i + 1, L):
            b = smooth[j]
            if b % a != 0:
                antichains[2].append((a, b))
    
    # Size 3: triples where no term divides another
    for i in range(L):
        a = smooth[i]
        for j in range(i + 1, L):
            b = smooth[j]
            if b % a == 0:
                continue
            for k in range(j + 1, L):
                c = smooth[k]
                if c % a == 0 or c % b == 0:
                    continue
                antichains[3].append((a, b, c))
    
    # Size 4: quadruples where no term divides another
    for i in range(L):
        a = smooth[i]
        for j in range(i + 1, L):
            b = smooth[j]
            if b % a == 0:
                continue
            for k in range(j + 1, L):
                c = smooth[k]
                if c % a == 0 or c % b == 0:
                    continue
                for m in range(k + 1, L):
                    d = smooth[m]
                    if d % a == 0 or d % b == 0 or d % c == 0:
                        continue
                    antichains[4].append((a, b, c, d))
    
    return antichains

def _find_min_terms(n, smooth, smooth_set, antichains):
    """Find the minimum number of distinct smooth terms summing to n,
    forming a divisibility antichain. Returns the list of terms or None."""
    candidates = [x for x in smooth if x <= n]
    
    # Try 1 term
    if n in smooth_set:
        return [n]
    
    # Try 2 terms using precomputed antichains
    for a, b in antichains[2]:
        if a + b == n and b <= n:
            return [a, b]
    
    # Try 3 terms using precomputed antichains
    for a, b, c in antichains[3]:
        if a + b + c == n and c <= n:
            return [a, b, c]
    
    # Try 4 terms using precomputed antichains
    for a, b, c, d in antichains[4]:
        if a + b + c + d == n and d <= n:
            return [a, b, c, d]
    
    # If 4 terms fail, try 5+ with DFS
    return _find_terms_dfs(n, smooth, 5)

def _find_terms_dfs(n, smooth, max_terms=8):
    """DFS search for minimum terms up to max_terms."""
    candidates = [x for x in smooth if x <= n]
    L = len(candidates)
    smooth_set = set(smooth)
    
    # Try from 5 to max_terms
    for k in range(5, max_terms + 1):
        res = _dfs_exact(n, candidates, k)
        if res is not None:
            return res
    
    return None

def _dfs_exact(n, candidates, k):
    """DFS to find exactly k terms summing to n with antichain property."""
    L = len(candidates)
    
    def dfs(start, remaining, current):
        if len(current) == k:
            if remaining == 0:
                return list(current)
            return None
        
        if remaining <= 0:
            return None
        
        # Pruning: need at least (k - len(current)) more terms, each >= candidates[start]
        if start >= L:
            return None
        
        min_possible = candidates[start] * (k - len(current))
        if remaining < min_possible:
            return None
        
        for i in range(start, L):
            x = candidates[i]
            if x > remaining:
                break
            
            # Check antichain with current
            valid = True
            for y in current:
                if y < x:
                    if x % y == 0:
                        valid = False
                        break
                else:
                    if y % x == 0:
                        valid = False
                        break
            
            if not valid:
                continue
            
            current.append(x)
            res = dfs(i + 1, remaining - x, current)
            current.pop()
            
            if res is not None:
                return res
        
        return None
    
    return dfs(0, n, [])

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    smooth = _generate_smooth(128)
    smooth_set = set(smooth)
    antichains = _precompute_antichains(smooth, max_size=4)
    
    answers = []
    total_terms = 0
    
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: use a simple valid solution
            # For n that is smooth, use [n]
            if n in smooth_set:
                answers.append([n])
                total_terms += 1
            else:
                # Try to find any valid representation quickly
                # Use 2 terms if possible
                found = False
                for a in smooth:
                    if a >= n:
                        break
                    b = n - a
                    if b > a and b in smooth_set and b % a != 0:
                        answers.append([a, b])
                        total_terms += 2
                        found = True
                        break
                if not found:
                    # Last resort: use 3 terms
                    for i in range(len(smooth)):
                        a = smooth[i]
                        if a >= n:
                            break
                        for j in range(i + 1, len(smooth)):
                            b = smooth[j]
                            if a + b >= n:
                                break
                            c = n - a - b
                            if c > b and c in smooth_set and b % a != 0 and c % a != 0 and c % b != 0:
                                answers.append([a, b, c])
                                total_terms += 3
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        # Should not happen, but return empty as last resort
                        answers.append([])
            continue
        
        res = _find_min_terms(n, smooth, smooth_set, antichains)
        
        if res is None:
            # This should not happen for n <= 128, but handle gracefully
            res = _find_terms_dfs(n, smooth, 10)
        
        if res is None:
            # Last resort
            res = []
        
        answers.append(res)
        total_terms += len(res)
        
    return answers

# EVOLVE-BLOCK-END
