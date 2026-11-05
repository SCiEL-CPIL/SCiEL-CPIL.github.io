"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _gcd_table(n):
    """Precompute gcd table for 1..n."""
    table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            table[i][j] = gcd(i, j)
            table[j][i] = gcd(i, j)
    return table

def _is_bad_triple(a, b, c, table):
    """Check if three elements have all equal pairwise gcds."""
    g_ab = table[a][b]
    g_ac = table[a][c]
    g_bc = table[b][c]
    return g_ab == g_ac == g_bc

def _count_bad_triples(S, table):
    """Count number of bad triples in set S."""
    count = 0
    for a, b, c in combinations(S, 3):
        if _is_bad_triple(a, b, c, table):
            count += 1
    return count

def _greedy_build(table, n=128):
    """Greedy construction: add elements that don't create bad triples."""
    S = []
    for x in range(1, n+1):
        ok = True
        for a, b in combinations(S, 2):
            if _is_bad_triple(a, b, x, table):
                ok = False
                break
        if ok:
            S.append(x)
    return S

def _local_search(S, table, n=128, deadline=None):
    """Local search with remove-and-refill swaps."""
    S_set = set(S)
    best_size = len(S)
    
    # Precompute which elements are "bad" with pairs in S
    # For each candidate x not in S, find how many pairs (a,b) in S would form bad triple with x
    def blocking_count(x):
        """Count pairs (a,b) in S such that (a,b,x) is a bad triple."""
        cnt = 0
        for a, b in combinations(S, 2):
            if _is_bad_triple(a, b, x, table):
                cnt += 1
        return cnt
    
    # Try removing each element and refilling
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        S_list = list(S)
        for i in range(len(S_list)):
            if deadline and time.time() > deadline:
                break
            removed = S_list[i]
            S_without = S_list[:i] + S_list[i+1:]
            S_without_set = set(S_without)
            
            # Try to add back elements not in S_without
            added = 0
            for x in range(1, n+1):
                if x in S_without_set:
                    continue
                # Check if x can be added to S_without
                ok = True
                for a, b in combinations(S_without, 2):
                    if _is_bad_triple(a, b, x, table):
                        ok = False
                        break
                if ok:
                    S_without.append(x)
                    S_without_set.add(x)
                    added += 1
            
            if len(S_without) > len(S_list):
                S = S_without
                S_set = set(S)
                best_size = len(S)
                improved = True
                break
            elif len(S_without) == len(S_list) and added > 0:
                # Same size but different composition - might help later
                # Only accept if it enables more additions
                pass
    
    return S, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s limit)
    
    n = 128
    table = _gcd_table(n)
    
    # Start with greedy
    S = _greedy_build(table, n)
    best_S = list(S)
    best_size = len(S)
    
    # Local search
    if time.time() < deadline:
        S, size = _local_search(S, table, n, deadline)
        if size > best_size:
            best_S = S
            best_size = size
    
    # Try multiple random restarts if time permits
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            # Random perturbation: start from greedy, then randomly remove some and refill
            S_restart = list(best_S)
            # Randomly remove 3-8 elements
            k = random.randint(3, 8)
            for _ in range(k):
                if S_restart:
                    idx = random.randint(0, len(S_restart)-1)
                    S_restart.pop(idx)
            
            S_restart_set = set(S_restart)
            # Try to add elements
            for x in range(1, n+1):
                if x in S_restart_set:
                    continue
                ok = True
                for a, b in combinations(S_restart, 2):
                    if _is_bad_triple(a, b, x, table):
                        ok = False
                        break
                if ok:
                    S_restart.append(x)
                    S_restart_set.add(x)
            
            if len(S_restart) > best_size:
                best_S = S_restart
                best_size = len(S_restart)
    
    return best_S
# EVOLVE-BLOCK-END
