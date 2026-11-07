import time
import sys
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i using bipartite matching."""
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        end = m + L
        if r > end:
            return False
        v = r
        while v <= end:
            adj[i - 1].append(v)
            v += i

    # Map right side values to indices for efficient matching
    val_to_idx = {}
    right_vals = []
    for i in range(n):
        for v in adj[i]:
            if v not in val_to_idx:
                val_to_idx[v] = len(right_vals)
                right_vals.append(v)

    # Convert adj to index-based
    adj_idx = [[val_to_idx[v] for v in adj[i]] for i in range(n)]

    # Order left nodes by number of candidates (most constrained first)
    order = sorted(range(n), key=lambda u: len(adj_idx[u]))

    match_r = [-1] * len(right_vals)
    sys.setrecursionlimit(10000)

    def dfs(u, visited):
        for v_idx in adj_idx[u]:
            if visited[v_idx]:
                continue
            visited[v_idx] = True
            if match_r[v_idx] == -1 or dfs(match_r[v_idx], visited):
                match_r[v_idx] = u
                return True
        return False

    for u in order:
        visited = [False] * len(right_vals)
        if not dfs(u, visited):
            return False
    return True

def _f(m, n=12):
    """Find smallest L in [1, 144] such that (m, m+L] has valid matching."""
    lo, hi = 1, 144
    if not _can_match(m, hi, n):
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 82  # 82s budget, 8s safety margin

    # LCM of 1..12 = 27720; f(m) is periodic with this period
    period = reduce(_lcm, range(1, 13))

    # Generate critical m values in [0, period-1] where f(m) could change:
    # m is critical if m+1 is a multiple of some i in 1..12
    critical_ms = set()
    for i in range(1, 13):
        for k in range(1, period // i + 1):
            m_val = k * i - 1
            if 0 <= m_val < period:
                critical_ms.add(m_val)
    critical_ms.add(0)

    critical_list = sorted(critical_ms)

    # Evaluate f at each critical point, track all that achieve the max
    best_f = 0
    best_residues = []

    for r in critical_list:
        if time.time() > deadline - 3:
            break
        f_val = _f(r)
        if f_val > best_f:
            best_f = f_val
            best_residues = [r]
        elif f_val == best_f:
            best_residues.append(r)

    # Find the largest m <= 1000000 for each best residue
    candidates = set()
    for r in best_residues:
        if r <= 1000000:
            k_max = (1000000 - r) // period
            mm = r + k_max * period
            candidates.add(mm)
            # Add neighbors for local refinement
            for delta in range(-5, 6):
                if 0 <= mm + delta <= 1000000:
                    candidates.add(mm + delta)

    # Also add the best_m from the first best residue as a fallback
    if best_residues:
        r0 = best_residues[0]
        if r0 <= 1000000:
            k_max = (1000000 - r0) // period
            mm0 = r0 + k_max * period
            for delta in range(-10, 11):
                if 0 <= mm0 + delta <= 1000000:
                    candidates.add(mm0 + delta)

    # If no candidates (shouldn't happen), use 0
    if not candidates:
        return 0

    # Local refinement: evaluate all candidates
    best_m = min(candidates)
    best_f_local = _f(best_m)

    for mm in sorted(candidates):
        if time.time() > deadline - 1:
            break
        f_val = _f(mm)
        if f_val > best_f_local:
            best_f_local = f_val
            best_m = mm

    # Final polish: check immediate neighbors of best_m
    if time.time() < deadline - 0.5:
        for delta in range(-3, 4):
            if time.time() > deadline - 0.3:
                break
            mm = best_m + delta
            if 0 <= mm <= 1000000:
                f_val = _f(mm)
                if f_val > best_f_local:
                    best_f_local = f_val
                    best_m = mm

    return int(best_m)
