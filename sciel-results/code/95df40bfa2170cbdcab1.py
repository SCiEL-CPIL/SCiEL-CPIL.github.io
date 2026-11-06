"""Rewritten program: maximum subset of [1,128] avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    n = 128
    # Precompute gcd matrix
    gcd_mat = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd(i, j)
    
    # Precompute for each pair (i,j), the set of k that would form a bad triple
    # A triple (a,b,c) is bad if gcd(a,b)==gcd(a,c)==gcd(b,c)==d
    # For a given pair (i,j) with gcd d, k is bad if gcd(i,k)==d and gcd(j,k)==d
    
    # Strategy: Use greedy + local search
    # Phase 1: Start with all numbers, then remove minimal set to eliminate bad triples
    # Phase 2: Try to add back numbers
    
    # Better: Build up greedily with smart ordering
    # Numbers that are "isolated" (primes, prime powers) are always safe
    
    # Let's identify which numbers can be in a bad triple
    # A number x can be in a bad triple if it shares a common gcd pattern with two others
    
    # Precompute: for each triple (i,j,k) with i<j<k, check if it's bad
    # This is O(n^3) = 128^3 ~ 2M, feasible
    
    # But we need maximum independent set in 3-uniform hypergraph
    # Use greedy: process numbers in order of "constraint degree"
    
    # For each number, count how many bad triples it's in
    bad_triple_count = [0]*(n+1)
    bad_triples = []  # list of (i,j,k) tuples that are bad
    
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            d = gcd_mat[i][j]
            for k in range(j+1, n+1):
                if gcd_mat[i][k] == d and gcd_mat[j][k] == d:
                    bad_triple_count[i] += 1
                    bad_triple_count[j] += 1
                    bad_triple_count[k] += 1
                    bad_triples.append((i, j, k))
    
    # Greedy: process numbers from least constrained to most constrained
    # But actually, for maximum independent set, we want to keep as many as possible
    # Use a greedy that tries to add each number if it doesn't conflict
    
    # Better approach: Start with all numbers, then iteratively remove numbers
    # that are in the most bad triples (greedy vertex cover in 3-uniform hypergraph)
    
    # Actually, let's try a different greedy: 
    # Sort by number of bad triples (ascending), add if compatible
    # This tends to keep "safe" numbers first
    
    # Precompute conflict structure using bitsets
    # For each number x, conflicts[x] = set of pairs (a,b) such that (x,a,b) is a bad triple
    # When we add x, we must not have both a and b already selected
    
    # More efficient: For each pair (a,b) that is selected, the set of x that would form bad triple
    # is determined by d=gcd(a,b): x must satisfy gcd(a,x)==d and gcd(b,x)==d
    
    # Let's precompute: for each pair (i,j), forbidden[i][j] = frozenset of k that make bad triple
    # But this is O(n^2 * n) memory... 128^2 * ~128 bits = manageable with bitsets
    
    # Use Python integers as bitsets
    # forbidden_pair[i][j] = bitmask of k values that form bad triple with i,j
    
    forbidden_pair = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            d = gcd_mat[i][j]
            mask = 0
            for k in range(1, n+1):
                if k != i and k != j and gcd_mat[i][k] == d and gcd_mat[j][k] == d:
                    mask |= (1 << k)
            forbidden_pair[i][j] = mask
            forbidden_pair[j][i] = mask
    
    # Greedy construction:
    # Process numbers in order of increasing "danger" (fewer bad triples first)
    # Try multiple orderings and keep best
    
    def try_greedy(order):
        selected = set()
        selected_mask = 0
        for x in order:
            if time.time() > deadline - 5:
                break
            # Check if adding x creates any bad triple with two already-selected numbers
            ok = True
            # For each pair (a,b) in selected, check if x is in forbidden_pair[a][b]
            # More efficient: for each a in selected, check if there exists b in selected 
            # such that x is forbidden by pair (a,b)
            # Equivalent: for each a in selected, the set of b in selected that conflict with x
            # is: b such that x in forbidden_pair[a][b]
            # i.e., b in selected AND (forbidden_pair[a][b] has bit x set)
            
            sel_list = list(selected)
            for idx_a, a in enumerate(sel_list):
                # For this a, find b in selected (after a in list) such that x is forbidden
                for b in sel_list[idx_a+1:]:
                    if (forbidden_pair[a][b] >> x) & 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.add(x)
                selected_mask |= (1 << x)
        return selected, selected_mask
    
    # Try several orderings
    best_set = set()
    best_mask = 0
    
    # Ordering 1: natural order
    order1 = list(range(1, n+1))
    s, m = try_greedy(order1)
    if len(s) > len(best_set):
        best_set, best_mask = s, m
    
    if time.time() > deadline - 5:
        return sorted(best_set) if best_set else [1]
    
    # Ordering 2: by bad_triple_count ascending (least constrained first)
    order2 = sorted(range(1, n+1), key=lambda x: bad_triple_count[x])
    s, m = try_greedy(order2)
    if len(s) > len(best_set):
        best_set, best_mask = s, m
    
    if time.time() > deadline - 5:
        return sorted(best_set) if best_set else [1]
    
    # Ordering 3: primes and prime powers first, then composites
    def is_prime_power(x):
        if x < 2:
            return False
        p = 2
        while p * p <= x:
            if x % p == 0:
                # Check if x is a power of p
                temp = x
                while temp % p == 0:
                    temp //= p
                return temp == 1
            p += 1
        return True  # x is prime
    
    order3 = sorted(range(1, n+1), key=lambda x: (not is_prime_power(x), x))
    s, m = try_greedy(order3)
    if len(s) > len(best_set):
        best_set, best_mask = s, m
    
    if time.time() > deadline - 5:
        return sorted(best_set) if best_set else [1]
    
    # Local search: try to add more numbers to best_set
    # For each number not in best_set, check if it can be added
    # If not, try removing one number from a conflicting triple and adding this one
    
    def can_add(x, sel_set, sel_mask):
        """Check if x can be added to sel_set without creating bad triple."""
        sel_list = list(sel_set)
        for idx_a, a in enumerate(sel_list):
            for b in sel_list[idx_a+1:]:
                if (forbidden_pair[a][b] >> x) & 1:
                    return False, None
        return True, None
    
    def try_local_improvement(sel_set, sel_mask):
        """Try to improve by swapping."""
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            sel_list = sorted(sel_set)
            for x in range(1, n+1):
                if time.time() > deadline - 3:
                    break
                if x in sel_set:
                    continue
                ok, _ = can_add(x, sel_set, sel_mask)
                if ok:
                    sel_set.add(x)
                    sel_mask |= (1 << x)
                    improved = True
                    break
            if not improved:
                # Try swaps: remove one, add one
                for x in range(1, n+1):
                    if time.time() > deadline - 3:
                        break
                    if x in sel_set:
                        continue
                    # Find minimal removal to allow x
                    # x conflicts with some pairs in sel_set
                    # Try removing one element from each conflicting pair
                    conflict_pairs = []
                    sel_list2 = sorted(sel_set)
                    for idx_a, a in enumerate(sel_list2):
                        for b in sel_list2[idx_a+1:]:
                            if (forbidden_pair[a][b] >> x) & 1:
                                conflict_pairs.append((a, b))
                    if not conflict_pairs:
                        continue
                    # Try removing one element from conflict_pairs to resolve all
                    # Simple: try removing each element that appears in conflict_pairs
                    candidates_to_remove = set()
                    for a, b in conflict_pairs:
                        candidates_to_remove.add(a)
                        candidates_to_remove.add(b)
                    for rem in candidates_to_remove:
                        if time.time() > deadline - 3:
                            break
                        new_set = sel_set - {rem}
                        ok, _ = can_add(x, new_set, 0)
                        if ok:
                            # Net zero change in size, but might enable further additions
                            # Only do if it doesn't reduce size (it doesn't)
                            # Actually this is neutral, skip for now
                            pass
        return sel_set, sel_mask
    
    best_set, best_mask = try_local_improvement(best_set, best_mask)
    
    if time.time() > deadline - 5:
        return sorted(best_set) if best_set else [1]
    
    # Try a different initial approach: start with all numbers, remove minimum
    # Greedy vertex cover for 3-uniform hypergraph
    remaining_bad = set(bad_triples)
    removed = set()
    
    while remaining_bad and time.time() < deadline - 10:
        # Find number in most remaining bad triples
        count_in_remaining = [0]*(n+1)
        for (i,j,k) in remaining_bad:
            count_in_remaining[i] += 1
            count_in_remaining[j] += 1
            count_in_remaining[k] += 1
        max_count = 0
        max_elem = -1
        for x in range(1, n+1):
            if x not in removed and count_in_remaining[x] > max_count:
                max_count = count_in_remaining[x]
                max_elem = x
        if max_elem == -1 or max_count == 0:
            break
        removed.add(max_elem)
        # Remove all triples containing max_elem
        remaining_bad = set(t for t in remaining_bad if max_elem not in t)
    
    candidate_from_removal = set(range(1, n+1)) - removed
    if len(candidate_from_removal) > len(best_set):
        best_set = candidate_from_removal
    
    return sorted(best_set) if best_set else [1]

# EVOLVE-BLOCK-END
