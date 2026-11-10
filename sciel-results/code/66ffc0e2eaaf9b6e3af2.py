"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
DEADLINE = 85.0  # safety margin before 90s

def _multiples_in_window(d, s, W):
    """Return list of multiples of d in [s, s+W]."""
    if s > MAX_VAL + W:
        return []
    lo = max(s, 1)
    hi = min(s + W, MAX_VAL)
    if lo > hi:
        return []
    first = ((lo + d - 1) // d) * d
    if first > hi:
        return []
    return list(range(first, hi + 1, d))

def _has_perfect_matching(candidates):
    """
    candidates: list of lists, candidates[i] = set of values available for divisor i.
    Returns True if there's a system of distinct representatives.
    Uses Hopcroft-Karp or simple DFS for small graphs.
    """
    n = len(candidates)
    # Build adjacency: for each divisor i, which slots (values) it can take
    # Values are global, so we need to map them to indices
    all_vals = set()
    for c in candidates:
        all_vals.update(c)
    val_to_idx = {v: i for i, v in enumerate(sorted(all_vals))}
    m = len(all_vals)
    
    # adj[i] = list of slot indices that divisor i can use
    adj = []
    for c in candidates:
        adj.append([val_to_idx[v] for v in c])
    
    # Try to find matching using DFS (augmenting path)
    match_r = [-1] * m  # which divisor is matched to each slot
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    for u in range(n):
        visited = [False] * m
        if not dfs(u, visited):
            return False
    return True

def construct():
    start_time = time.time()
    
    # Try spans from minimum possible upward
    # Minimum span for 10 distinct positive integers is 9
    for W in range(9, 200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try different starting positions s
        # s must be >= 1, and s+W <= MAX_VAL
        # For small W, try s from 1 to some reasonable bound
        # The pattern repeats with period lcm, but that's large.
        # Instead, try s in a range that covers different residue classes.
        # For each divisor d, the relevant s values are those where [s, s+W] contains a multiple of d.
        # We can try s = 1, 2, ..., up to some limit.
        
        found = False
        # Try s from 1 to max(s_max, some bound)
        # For small W, the "interesting" s values are near multiples of the divisors.
        # Let's try s from 1 to 5000, and also s near MAX_VAL - W
        s_candidates = set()
        for s in range(1, min(5000, MAX_VAL - W + 1)):
            s_candidates.add(s)
        # Also try large s
        for s in range(max(1, MAX_VAL - W - 500), MAX_VAL - W + 1):
            s_candidates.add(s)
        
        for s in sorted(s_candidates):
            if time.time() - start_time > DEADLINE:
                return None  # Should not happen if we have a fallback
            
            if s < 1 or s + W > MAX_VAL:
                continue
            
            candidates = []
            feasible = True
            for d in DIVISORS:
                cands = _multiples_in_window(d, s, W)
                if not cands:
                    feasible = False
                    break
                candidates.append(cands)
            
            if not feasible:
                continue
            
            if _has_perfect_matching(candidates):
                # Extract the actual assignment
                # Re-run matching to get the assignment
                all_vals = set()
                for c in candidates:
                    all_vals.update(c)
                val_to_idx = {v: i for i, v in enumerate(sorted(all_vals))}
                m = len(all_vals)
                adj = []
                for c in candidates:
                    adj.append([val_to_idx[v] for v in c])
                
                match_r = [-1] * m
                match_l = [-1] * len(DIVISORS)
                
                def dfs2(u, visited):
                    for v in adj[u]:
                        if visited[v]:
                            continue
                        visited[v] = True
                        if match_r[v] == -1 or dfs2(match_r[v], visited):
                            match_r[v] = u
                            match_l[u] = v
                            return True
                    return False
                
                for u in range(len(DIVISORS)):
                    visited = [False] * m
                    dfs2(u, visited)
                
                result = [sorted(all_vals)[match_l[u]] for u in range(len(DIVISORS))]
                # Verify
                assert all(isinstance(x, int) for x in result)
                assert all(1 <= x <= MAX_VAL for x in result)
                assert all(result[i] % DIVISORS[i] == 0 for i in range(len(DIVISORS)))
                assert len(set(result)) == len(result)
                return result
        
        # If no s worked for this W, try next W
    
    # Fallback: return divisors themselves
    return DIVISORS[:]

# EVOLVE-BLOCK-END
