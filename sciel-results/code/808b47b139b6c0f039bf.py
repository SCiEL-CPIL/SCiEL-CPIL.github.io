"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _has_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i|b_i."""
    lo = m + 1
    hi = m + L
    if hi < lo:
        return False
    
    # For each i in 1..n, find which numbers in (m, m+L] are divisible by i
    valid = {}
    for i in range(1, n + 1):
        first = ((lo + i - 1) // i) * i
        if first > hi:
            return False
        multiples = []
        x = first
        while x <= hi:
            multiples.append(x)
            x += i
        valid[i] = multiples
    
    # Map right side to indices
    all_numbers = set()
    for i in range(1, n + 1):
        all_numbers.update(valid[i])
    
    num_to_idx = {x: j for j, x in enumerate(sorted(all_numbers))}
    num_nodes = len(all_numbers)
    
    if num_nodes < n:
        return False
    
    # Adjacency: for each left node i, which right nodes are adjacent
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        for x in valid[i]:
            adj[i-1].append(num_to_idx[x])
    
    # Bipartite matching via DFS (Kuhn's algorithm)
    match_r = [-1] * num_nodes
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    match_count = 0
    for u in range(n):
        visited = [False] * num_nodes
        if dfs(u, visited):
            match_count += 1
    
    return match_count == n


def _compute_f(m, n=12, max_L=200):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    for L in range(1, max_L + 1):
        if _has_matching(m, L, n):
            return L
    return max_L


def construct():
    """Return m in 0..1000000 that maximizes f(12, m)."""
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # f(12, m) is periodic with period lcm(1..12) = 2520
    # Check all residues mod 2520 to find the best m
    best_m = 0
    best_L = 0
    
    for r in range(2520):
        if time.time() > deadline:
            break
        L = _compute_f(r, n=12, max_L=150)
        if L > best_L:
            best_L = L
            best_m = r
    
    return int(best_m)
# EVOLVE-BLOCK-END
