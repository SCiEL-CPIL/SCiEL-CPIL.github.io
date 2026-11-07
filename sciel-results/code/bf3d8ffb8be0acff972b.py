"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from functools import lru_cache

P = {'n': 12, 'maximum_start': 1000000}

# Precompute divisors for 1..12
DIVISORS = {}
for i in range(1, 13):
    DIVISORS[i] = [d for d in range(1, i + 1) if i % d == 0]

def _can_match(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i | b_i."""
    n = 12
    # For each i, the candidate b_i must be in (m, m+L] and divisible by i
    # b_i = k*i for some k, with m < k*i <= m+L
    # So k in (m/i, (m+L)/i]
    
    # Build bipartite graph: left = {1..12}, right = candidate numbers
    # But we need distinct assignment. Use DFS with backtracking (Hungarian-like).
    
    # For each i, list of possible b_i values
    candidates = {}
    for i in range(1, 13):
        # b_i must be divisible by i, in (m, m+L]
        # smallest multiple of i > m: ceil((m+1)/i) * i
        # largest multiple of i <= m+L: floor((m+L)/i) * i
        lo_k = (m + 1 + i - 1) // i  # ceil((m+1)/i)
        hi_k = (m + L) // i
        if lo_k > hi_k:
            return False
        # candidates[i] = [lo_k*i, (lo_k+1)*i, ..., hi_k*i]
        # But we need to ensure distinctness across all i
        # Store as list of actual values
        vals = []
        for k in range(lo_k, hi_k + 1):
            v = k * i
            if m < v <= m + L:
                vals.append(v)
        if not vals:
            return False
        candidates[i] = vals
    
    # Now we need a system of distinct representatives (SDR)
    # Use DFS with backtracking, trying to assign each i a unique value
    # Order i by number of candidates (ascending) for efficiency
    
    order = sorted(range(1, 13), key=lambda i: len(candidates[i]))
    
    used = set()
    
    def dfs(idx):
        if idx == 12:
            return True
        i = order[idx]
        for v in candidates[i]:
            if v not in used:
                used.add(v)
                if dfs(idx + 1):
                    return True
                used.remove(v)
        return False
    
    return dfs(0)

def _f_val(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct b_1..b_12 with i|b_i."""
    # Binary search on L
    # Lower bound: at least 12 (need 12 distinct numbers)
    # Upper bound: 144 per task description
    lo = 1
    hi = 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    
    best_m = 0
    best_f = 0
    
    # First, evaluate a set of strategic candidates
    # Strategy: m values that create "gaps" in divisibility coverage
    # For each i, the interval (m, m+L] must contain a multiple of i.
    # If m is chosen so that for some i, the next multiple of i after m is far away,
    # then L must be large enough to reach that multiple.
    
    # Key insight: for a given m, the constraint for i is that there exists k such that
    # m < k*i <= m+L. The tightest constraint is the one where the gap to the next
    # multiple of i is largest.
    
    # For each i, the "gap" is: next_mult(i, m) - m, where next_mult(i, m) = smallest multiple of i > m
    # = ((m // i) + 1) * i
    # So gap_i = ((m // i) + 1) * i - m = i - (m % i) if m % i != 0, else i
    
    # But we also need distinctness, which can force L to be even larger.
    
    # Let's search over m values.
    
    # Phase 1: Random sampling
    random.seed(42)
    
    # Generate candidate m values
    candidates = set()
    
    # Add some structured candidates
    # m = k * lcm(1..12) - 1 = k*27720 - 1, which makes m+1 divisible by all i
    # Actually m = k*27720 - 1 means m % i = i-1 for all i, so gap_i = 1 for all i. That's bad (small L).
    # We want m % i to be small (close to 0) for some i, making gap_i large.
    # gap_i = i - (m % i) if m % i != 0, else i
    # To maximize min_i gap_i, we want m % i to be small for all i.
    # m % i = 0 for all i means m is multiple of lcm(1..12) = 27720.
    # Then gap_i = i for all i, so L >= 12.
    
    # But distinctness may force larger L.
    
    # Let's try m values near multiples of 27720
    for k in range(0, 37):  # 0*27720 to 36*27720, all <= 1000000
        m_val = k * 27720
        if 0 <= m_val <= 1000000:
            candidates.add(m_val)
            # Also try m_val - 1, m_val + 1, etc.
            for offset in range(-5, 6):
                m2 = m_val + offset
                if 0 <= m2 <= 1000000:
                    candidates.add(m2)
    
    # Add random candidates
    for _ in range(5000):
        m_val = random.randint(0, 1000000)
        candidates.add(m_val)
    
    # Add candidates based on maximizing individual gaps
    # For each i, find m where m % i is small
    for i in range(2, 13):
        for target_mod in range(0, min(i, 5)):
            # m ≡ target_mod (mod i), m in [0, 1000000]
            # m = target_mod + k*i
            # Find a few such m
            start = target_mod
            count = 0
            for k in range(0, 1000000 // i + 1):
                m_val = target_mod + k * i
                if m_val > 1000000:
                    break
                if count < 20:
                    candidates.add(m_val)
                count += 1
                if count >= 20:
                    break
    
    # Evaluate candidates
    for m_val in sorted(candidates):
        if time.time() > deadline - 10:
            break
        f = _f_val(m_val)
        if f > best_f:
            best_f = f
            best_m = m_val
    
    # Phase 2: Local search around best_m
    if time.time() < deadline - 5:
        # Try neighbors of best_m
        for delta in range(-20, 21):
            m_val = best_m + delta
            if 0 <= m_val <= 1000000 and m_val != best_m:
                if time.time() > deadline - 3:
                    break
                f = _f_val(m_val)
                if f > best_f:
                    best_f = f
                    best_m = m_val
    
    # Phase 3: More targeted search if time remains
    if time.time() < deadline - 3:
        # Try m values that maximize the minimum gap
        # For a given m, min_gap = min_i (i - m % i if m % i != 0 else i)
        # We want to maximize this, but distinctness matters too.
        
        # Try a grid of m values
        step = 100
        for m_val in range(0, 1000001, step):
            if time.time() > deadline - 1:
                break
            f = _f_val(m_val)
            if f > best_f:
                best_f = f
                best_m = m_val
    
    return int(best_m)

# EVOLVE-BLOCK-END
