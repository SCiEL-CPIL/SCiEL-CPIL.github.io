"""Improved program: greedy + local search with bitset acceleration and deadline guard."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
from itertools import combinations

def construct():
    """
    Returns a list of distinct positive integers <= 192 (max 48) such that
    no selected integer divides the sum of any 3 distinct OTHER selected integers.
    Maximizes cardinality.
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 192
    MAX_K = 48
    
    # Precompute all triples of distinct numbers from 1..192
    # For each number a in 1..192, we need to check if a divides sum of 3 other selected numbers
    
    # Strategy:
    # 1. Greedy construction: add numbers in some order, checking validity
    # 2. Local search: try to swap or add/remove to improve
    
    # For efficiency, we'll use a bitset approach where possible
    
    # Precompute for each pair (i, j) and each potential a, whether a | (i + j + k) for some k
    # Actually, let's think differently:
    # For a candidate set S, for each a in S, we need: for all distinct b, c, d in S \ {a}, a does not divide (b + c + d)
    
    # Let's precompute: for each a in 1..192, and for each pair (b, c) with b != c, b != a, c != a,
    # what values of d (d != a, d != b, d != c) would make a | (b + c + d)?
    # That is: d ≡ -(b + c) (mod a), with d in 1..192, d != a, d != b, d != c
    
    # This is complex. Let's use a different approach:
    # For a given set S, check validity by iterating over a in S, then over triples in S\{a}
    # But this is O(|S|^4) which for |S|=48 is about 48*47*46*45/6 ≈ 1.8M operations per check, which is slow
    
    # Better: for each a in S, precompute the set of "forbidden" sums: 
    #   forbidden_sums[a] = { (b+c+d) mod a == 0 for distinct b,c,d in S\{a} }
    # But we need to check if ANY triple sums to a multiple of a.
    
    # Alternative: for each a, consider all pairs (b,c) in S\{a}, b<c, then for each d in S\{a,b,c},
    # check if (b+c+d) % a == 0. Still O(|S|^3) per a, O(|S|^4) total.
    
    # For |S| up to 48, |S|^4 ≈ 5.3M, which might be borderline in Python.
    # Let's try to optimize with early termination and bitsets.
    
    # Actually, let's use a smarter approach:
    # For a fixed a, and a set T = S \ {a}, we want to know if there exist distinct b, c, d in T
    # such that (b + c + d) % a == 0.
    # This is equivalent to: does T contain a 3-element subset whose sum is ≡ 0 (mod a)?
    
    # We can precompute for each a and each residue class mod a, the count of elements in T with that residue.
    # Then use combinatorics to check if a zero-sum triple exists.
    
    # But this is still complex. Let's just do the direct check with some optimizations.
    
    def is_valid_set(S):
        """Check if set S satisfies the condition."""
        S_list = list(S)
        n = len(S_list)
        for a in S_list:
            # Check all triples in S \ {a}
            others = [x for x in S_list if x != a]
            m = len(others)
            if m < 3:
                continue
            # For each triple (b, c, d) in others, check if (b+c+d) % a == 0
            # Optimize: for each pair (b, c), check if there exists d such that (b+c+d) % a == 0
            # i.e., d % a == (-b-c) % a
            # Precompute residue counts for others
            from collections import defaultdict
            residue_count = defaultdict(int)
            for x in others:
                residue_count[x % a] += 1
            
            # For each pair (b, c) with b < c in others:
            # We need d in others, d != b, d != c, such that d % a == (-b-c) % a
            # Count how many such d exist
            for i in range(m):
                b = others[i]
                for j in range(i+1, m):
                    c = others[j]
                    target_res = (-b - c) % a
                    # Count of elements in others with residue target_res, minus b and c if they have that residue
                    cnt = residue_count[target_res]
                    if b % a == target_res:
                        cnt -= 1
                    if c % a == target_res:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True
    
    # Greedy construction
    def greedy_construct(order):
        S = []
        for x in order:
            if x in S:
                continue
            S_new = S + [x]
            if is_valid_set(S_new):
                S = S_new
            if len(S) >= MAX_K:
                break
        return S
    
    # Local search: try to improve by swapping
    def local_search(S, remaining_time):
        S_set = set(S)
        improved = True
        while improved and time.time() - start_time < remaining_time:
            improved = False
            # Try to add any number not in S
            for x in range(1, N+1):
                if x in S_set:
                    continue
                S_new = list(S) + [x]
                if is_valid_set(S_new):
                    S = S_new
                    S_set.add(x)
                    improved = True
                    break
            if time.time() - start_time > remaining_time:
                break
            # Try to swap: remove one, add one
            for i, x in enumerate(S):
                if time.time() - start_time > remaining_time:
                    break
                S_without = S[:i] + S[i+1:]
                for y in range(1, N+1):
                    if y in S_set:
                        continue
                    S_new = S_without + [y]
                    if is_valid_set(S_new):
                        if len(S_new) > len(S):
                            S = S_new
                            S_set = set(S_new)
                            improved = True
                            break
                if improved:
                    break
        return S
    
    # Initial greedy with descending order
    best_S = greedy_construct(range(N, 0, -1))
    
    # Try different orders
    if time.time() - start_time < DEADLINE:
        # Ascending
        S2 = greedy_construct(range(1, N+1))
        if len(S2) > len(best_S):
            best_S = S2
    
    if time.time() - start_time < DEADLINE:
        # Random orders
        random.seed(42)
        for _ in range(100):
            if time.time() - start_time > DEADLINE:
                break
            order = list(range(1, N+1))
            random.shuffle(order)
            S3 = greedy_construct(order)
            if len(S3) > len(best_S):
                best_S = S3
    
    # Local search on best_S
    if time.time() - start_time < DEADLINE:
        best_S = local_search(best_S, DEADLINE)
    
    return best_S

# EVOLVE-BLOCK-END
