"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-END

# --- Core logic ---

def _max_matching(bipartite):
    """
    bipartite: list of lists; bipartite[i] is the set of right-nodes adjacent to left-node i.
    Returns the size of the maximum matching.
    """
    n = len(bipartite)
    match_r = {}
    def dfs(u, seen):
        for v in bipartite[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False
    res = 0
    for u in range(n):
        if dfs(u, set()):
            res += 1
    return res

def _f_12(m):
    """
    Computes f(12, m): the least positive L such that (m, m+L] contains
    distinct b_1, ..., b_12 with i dividing b_i.
    Returns 0 if no such L exists (should not happen for reasonable L).
    """
    # We search for the minimum L.
    # The max possible L is bounded. For n=12, the "worst" case is usually small.
    # Let's binary search or linear search?
    # The task says "Maximize this independently computed L". Wait.
    # "f(12,m), the LEAST positive L such that ... Maximize this independently computed L."
    # So we want to find m that maximizes f(12,m).
    # f(12,m) is the length of the "gap" before we can find a valid matching.
    
    # Let's just iterate L from 1 upwards until we find a matching.
    # Since the max L is at most 144 (per task description), this is fast.
    
    # For a given L, the candidates for b_i are in (m, m+L].
    # b_i must be divisible by i.
    # So b_i is a multiple of i in that range.
    
    # We can construct the bipartite graph:
    # Left nodes: 1..12
    # Right nodes: the distinct integers in (m, m+L]
    # Edge (i, x) exists if i divides x.
    # We need a perfect matching.
    
    # Optimization:
    # The set of right nodes is fixed for a given m and L.
    # Actually, the right nodes are just the integers in (m, m+L].
    # There are L such integers.
    # We need to match 12 specific divisors to 12 distinct numbers.
    
    # Let's precompute the multiples for each i in the range (m, m+L].
    # For a fixed L, the set of numbers is {m+1, ..., m+L}.
    
    # We'll iterate L from 1 to 144 (or until found).
    # For each L, check if a perfect matching exists.
    
    max_L = 144 # Upper bound from task
    for L in range(1, max_L + 1):
        # Build adjacency list for left nodes 1..12
        # Right nodes are the numbers m+1 .. m+L.
        # Let's index right nodes by 0..L-1 corresponding to m+1..m+L
        
        # Instead of building the full graph every time, we can be smart.
        # But L is small (<=144), and 12 left nodes.
        # Building the graph is O(12 * L).
        
        adj = [[] for _ in range(12)]
        # Map number to index? Or just use the number itself as the right node ID?
        # Using the number itself is easier for the matching algorithm if we adjust it.
        # Let's use the actual numbers as right nodes.
        
        # Reset matching state
        # We need a matching from {1..12} to {m+1..m+L}
        
        # Let's use the standard Hopcroft-Karp or simple DFS for small graphs.
        # Simple DFS (Kuhn's algorithm) is O(V*E). V=12+L, E=12*L. Very fast.
        
        # Right nodes are the integers in the range.
        # We'll store match_r: right_node -> left_node
        
        # To avoid recreating the adjacency list from scratch if possible?
        # No, the range changes with L.
        
        # Let's just run the matching check.
        
        # Optimization: Pre-compute which numbers in (m, m+L] are divisible by which i.
        # Actually, for a specific L, the set of numbers is fixed.
        
        # Let's implement the check inside the loop.
        
        # We need to check if there is a matching covering all left nodes 1..12.
        
        # Let's use a helper that takes m and L and returns True/False.
        
        if _has_matching(m, L):
            return L
            
    return 0 # Should not reach here

def _has_matching(m, L):
    """
    Checks if (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    # Right nodes: m+1, ..., m+L
    # Left nodes: 1, ..., 12
    
    # Build adjacency: left i -> list of right nodes v in (m, m+L] such that v % i == 0
    
    adj = [[] for _ in range(12)]
    
    for v in range(m + 1, m + L + 1):
        for i in range(1, 13):
            if v % i == 0:
                adj[i-1].append(v)
                
    # Now run bipartite matching
    # Left: 0..11
    # Right: integers m+1..m+L
    
    match_r = {} # right_node -> left_node
    
    def dfs(u, seen):
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False
        
    count = 0
    for u in range(12):
        if dfs(u, set()):
            count += 1
            
    return count == 12

# --- Precomputation and Search ---

# The function f(12, m) depends on m modulo the LCM of 1..12.
# LCM(1..12) = 27720.
# Let P = 27720.
# f(12, m) = f(12, m % P) ?
# Let's verify.
# The condition is about divisibility by 1..12.
# If we shift m by P, the residues modulo 1..12 of the numbers in the interval shift.
# Specifically, the set of numbers in (m, m+L] is {m+1, ..., m+L}.
# The set of numbers in (m+P, m+P+L] is {m+P+1, ..., m+P+L}.
# Note that (m+P+k) % i = (m+k) % i for all i in 1..12.
# So the divisibility structure is identical.
# The distinctness constraint is also preserved (just shifted).
# So yes, f(12, m) depends only on m % 27720.

P = 27720

def _compute_f_for_residues():
    """
    Computes f(12, r) for r in 0..P-1.
    Returns a list of length P.
    """
    results = [0] * P
    # We can parallelize? No, standard library only.
    # We can optimize by noting that f(12, m) is small.
    # Let's just iterate.
    
    # To speed up, we can precompute the "next multiple" logic?
    # Or just call _has_matching.
    # _has_matching is O(L * 12 * matching_cost).
    # Matching cost is small.
    # Total ops: P * 144 * 12 * 12 ~ 27720 * 20736 ~ 500 million?
    # That might be too slow in Python.
    
    # Let's optimize _has_matching.
    # Instead of building the adjacency list from scratch for each L,
    # we can precompute the divisibility of numbers in a window?
    
    # Alternative:
    # For a fixed residue r, we want to find the min L.
    # We can simulate the interval growing.
    
    # Let's try to optimize the matching check.
    # The right nodes are m+1..m+L.
    # We can represent the right nodes by their offset 1..L.
    # But the divisibility depends on the actual value m+offset.
    
    # Let's precompute for each residue r, the sequence of residues of r+1, r+2, ...
    # Actually, we can just iterate L and check.
    
    # To speed up Python:
    # 1. Use local variables.
    # 2. Avoid function call overhead in the inner loop if possible.
    # 3. Use a faster matching algorithm? Hopcroft-Karp is O(E*sqrt(V)), but V is small.
    #    Simple DFS is fine.
    
    # Let's rewrite _has_matching to be inline or as fast as possible.
    
    # Actually, 27720 * 144 is about 4 million checks.
    # Each check involves building a graph of size 12xL and running matching.
    # If each check takes ~10 microseconds, total is 40 seconds.
    # We have 90 seconds. It should fit.
    
    # Let's add a time guard.
    
    start_time = time.time()
    deadline = start_time + 80 # Leave 10s margin
    
    for r in range(P):
        if time.time() > deadline:
            break
        # Compute f(12, r)
        # Note: m=r. The interval is (r, r+L].
        # If r=0, interval is (0, L] -> 1..L.
        
        # We can cache results? No, each r is unique.
        
        # Optimization:
        # The matching only depends on the residues of the numbers modulo 1..12.
        # But we already use the actual values for divisibility.
        
        # Let's just call _has_matching.
        
        # We can early exit if we find a small L? No, we need the min L.
        
        # Let's try to optimize _has_matching by passing m and L.
        
        # Inline the matching logic for speed?
        
        # Let's stick to _has_matching but make it tight.
        
        # One optimization:
        # If L < 12, it's impossible to have 12 distinct numbers.
        # So start L from 12.
        
        found = False
        for L in range(12, 145): # Max 144
            if _has_matching(r, L):
                results[r] = L
                found = True
                break
        if not found:
            results[r] = 0 # Should not happen
            
    return results

# --- Main Construct ---

def construct():
    # We want to maximize f(12, m).
    # Since f(12, m) depends only on m % P, we just need to find
    # the residue r that maximizes f(12, r).
    # Then we can return any m in [0, 1000000] such that m % P == r.
    # The smallest such m is r itself (if r <= 1000000, which is true).
    
    # So the task reduces to:
    # 1. Compute f(12, r) for all r in 0..P-1.
    # 2. Find r_max = argmax f(12, r).
    # 3. Return r_max.
    
    # Let's do this.
    
    # To ensure we finish in time, we'll compute all residues.
    # If we run out of time, we return the best found so far.
    
    # However, the computation of all residues is deterministic and should finish.
    # Let's verify the speed.
    
    # If it's too slow, we can sample? No, we need the global max.
    # But the period is small (27720).
    
    # Let's run the precomputation.
    
    # We'll store the best m and best value.
    
    # To be safe, we'll compute in chunks or just run it.
    
    # Let's run it.
    
    # Optimization:
    # We can precompute the divisibility for a sliding window?
    # Not really, since m changes.
    
    # Let's just run the loop.
    
    # We'll use a global cache for matching results?
    # The matching depends on (m, L).
    # m % P is unique for each r.
    # So no cache hits between different r.
    
    # Let's execute.
    
    # To handle the 90s limit, we'll time it.
    
    start = time.time()
    
    # We'll compute the values.
    # If we can't finish all, we'll return the best from the completed part.
    # But 27720 is small.
    
    # Let's try to optimize _has_matching further.
    
    # Idea:
    # The set of right nodes is {m+1, ..., m+L}.
    # We can precompute the list of numbers and their divisibility by 1..12.
    
    # Let's rewrite _has_matching to be faster.
    
    # Actually, let's just run the current version and see if it's fast enough.
    # If not, we'll optimize.
    
    # For now, let's assume it's fast enough.
    
    # We'll compute the array of f-values.
    
    # To avoid global state issues, we'll do it locally.
    
    # But _has_matching is global.
    
    # Let's call _compute_f_for_residues.
    
    # Wait, _compute_f_for_residues has a deadline.
    # Let's pass the deadline to it or handle it here.
    
    # Let's just call it.
    
    # But if it times out, it returns a partial list?
    # My _compute_f_for_residues breaks and returns a list with 0s for uncomputed.
    # That's dangerous. We should return the best found.
    
    # Let's modify the strategy:
    # Iterate r from 0 to P-1.
    # Compute f(12, r).
    # Track max_f and best_r.
    # If time runs out, return best_r.
    
    # This is safer.
    
    best_r = 0
    best_f = 0
    
    # We'll use a local deadline.
    deadline = time.time() + 85
    
    for r in range(P):
        if time.time() > deadline:
            break
            
        # Compute f(12, r)
        # Start L from 12
        f_val = 0
        for L in range(12, 145):
            if _has_matching(r, L):
                f_val = L
                break
                
        if f_val > best_f:
            best_f = f_val
            best_r = r
            
    return int(best_r)

# EVOLVE-BLOCK-START
# (The block above is the implementation)
# EVOLVE-BLOCK-END
