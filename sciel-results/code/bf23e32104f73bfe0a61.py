"""Rewritten program: stochastic local search for max subset of [1,128] with no equal-gcd triples."""
import time
import random
from math import gcd
from itertools import combinations

# Precompute all GCD triples that are "bad" (all three pairwise gcds equal)
# A triple {a,b,c} is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)

def _precompute():
    """Precompute conflict structure."""
    n = 128
    # gcd table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # For each pair (a,b), find all c > b such that {a,b,c} is a bad triple
    # bad triple: gcd(a,b) == gcd(a,c) == gcd(b,c)
    # This means gcd(a,b) = gcd(a,c) = gcd(b,c) = d
    # So a, b, c are all multiples of d, and a/d, b/d, c/d are pairwise coprime
    # Actually more precisely: gcd(a,b)=d, gcd(a,c)=d, gcd(b,c)=d
    # This means a = d*a', b = d*b', c = d*c' where gcd(a',b')=gcd(a',c')=gcd(b',c')=1
    
    # conflicts[i] = set of pairs (j,k) with j<k, j,k != i, such that {i,j,k} is bad
    # Actually, let's think of it as: for each element i, which pairs (j,k) form a bad triple with i
    # Better: for each element i, conflicts[i] is a set of frozensets or a bitset of elements that are "conflicted"
    # Actually the constraint is on triples. Let's use a different approach.
    
    # For local search, we need to quickly check if adding element x to set S creates a bad triple.
    # Adding x creates a bad triple iff there exist a,b in S such that gcd(a,b)==gcd(a,x)==gcd(b,x).
    
    # Precompute for each x and each pair (a,b): is {a,b,x} bad?
    # That's O(n^3) = 128^3 ≈ 2M, which is fine.
    
    # Let's precompute bad_triples[x] = set of (a,b) with a<b, a!=x, b!=x, such that {a,b,x} is bad
    # Actually for efficiency in local search, let's precompute:
    # For each x, bad_pairs_with_x[x] = set of (a,b) such that gcd(a,b)==gcd(a,x)==gcd(b,x)
    
    bad_pairs_with = [set() for _ in range(n+1)]
    for x in range(1, n+1):
        for a in range(1, x):
            gax = gcd(a, x)
            for b in range(a+1, x):
                if gcd(a,b) == gax and gcd(b,x) == gax:
                    bad_pairs_with[x].add((a,b))
                    bad_pairs_with[a].add((b,x))
                    bad_pairs_with[b].add((a,x))
        # Also need to handle pairs where one is > x
        # Actually let's just do all pairs
    # Redo more carefully
    bad_pairs_with = [set() for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            gab = gcd(a, b)
            for c in range(b+1, n+1):
                if gcd(a,c) == gab and gcd(b,c) == gab:
                    bad_pairs_with[a].add((b,c))
                    bad_pairs_with[b].add((a,c))
                    bad_pairs_with[c].add((a,b))
    
    return gcd_table, bad_pairs_with

# Precompute at module level
_gcd_table, _bad_pairs_with = _precompute()

def _is_valid_set(S):
    """Check if set S has no bad triples."""
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a, b = S_list[i], S_list[j]
            if (a,b) in _bad_pairs_with or (b,a) in _bad_pairs_with:
                # Need to check if there's a third element in S forming a bad triple
                # Actually, _bad_pairs_with[a] contains (b,c) if {a,b,c} is bad
                # So if b is in S and (a,b) is in bad_pairs_with[c] for some c in S...
                # Let me rethink.
                pass
    # Better check:
    for a in S_list:
        for (b,c) in _bad_pairs_with[a]:
            if b in S and c in S:
                return False
    return True

def _count_violations(S):
    """Count number of bad triples in S."""
    count = 0
    S_set = set(S)
    for a in S:
        for (b,c) in _bad_pairs_with[a]:
            if b in S_set and c in S_set:
                count += 1
    # Each bad triple is counted 3 times (once for each element)
    return count // 3

def _greedy_build(order=None, deadline=None):
    """Greedy build with given order."""
    if order is None:
        order = list(range(1, 129))
    S = set()
    S_set = set()
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        # Check if adding x creates any bad triple
        can_add = True
        for (a,b) in _bad_pairs_with[x]:
            if a in S_set and b in S_set:
                can_add = False
                break
        if can_add:
            S.add(x)
            S_set.add(x)
    return S

def _local_search(S, deadline, max_iterations=50000):
    """Local search: 1-swap and 1-add moves."""
    S_set = set(S)
    best_S = set(S)
    best_size = len(best_S)
    current_size = len(S)
    
    # Precompute: for each x not in S, which pairs in S would conflict
    # For efficiency, maintain conflict counts
    
    # For each element y in S, maintain count of how many bad triples it's part of with other elements in S
    # Actually, let's just do simple local search
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try 1-add: add an element not in S
        best_gain = 0
        best_add = None
        for x in range(1, 129):
            if x in S_set:
                continue
            # Check if adding x is valid
            can_add = True
            for (a,b) in _bad_pairs_with[x]:
                if a in S_set and b in S_set:
                    can_add = False
                    break
            if can_add:
                if 1 > best_gain:
                    best_gain = 1
                    best_add = x
        
        if best_add is not None:
            S_set.add(best_add)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_S = set(S_set)
            continue
        
        # Try 1-swap: remove y from S, add x not in S
        best_swap = None
        best_swap_gain = 0
        # Only try swaps if we might improve (i.e., we're stuck)
        for y in list(S_set):
            # Temporarily remove y
            S_set.remove(y)
            for x in range(1, 129):
                if x in S_set:
                    continue
                # Check if adding x is valid with y removed
                can_add = True
                for (a,b) in _bad_pairs_with[x]:
                    if a in S_set and b in S_set:
                        can_add = False
                        break
                if can_add:
                    # This swap is valid (net change 0, but might enable future adds)
                    # Actually for pure cardinality, a swap doesn't help unless it enables adds
                    # Let's just do: swap y for x if x is "better" in some sense
                    # For now, just accept any valid swap to escape local minima
                    if best_swap is None:
                        best_swap = (y, x)
                        break
            S_set.add(y)
            if best_swap is not None:
                break
        
        if best_swap is not None:
            y, x = best_swap
            S_set.remove(y)
            S_set.add(x)
            # Size stays the same, but we might be able to add more later
            continue
        
        # No move found, break
        break
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_set = set()
    best_size = 0
    
    # Multiple random restarts
    rng = random.Random(42)
    
    # First, try a deterministic greedy with different orderings
    # Order by: primes first, then by number of prime factors, etc.
    
    # Strategy 1: Greedy with natural order
    S = _greedy_build(list(range(1, 129)), deadline)
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    # Strategy 2: Greedy with reverse order
    if time.time() < deadline:
        S = _greedy_build(list(range(128, 0, -1)), deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 3: Greedy with primes first
    if time.time() < deadline:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5)+1):
                if n % i == 0:
                    return False
            return True
        
        primes = [x for x in range(1, 129) if is_prime(x)]
        composites = [x for x in range(1, 129) if not is_prime(x)]
        S = _greedy_build(primes + composites, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 4: Local search from greedy solutions
    if time.time() < deadline:
        S = _greedy_build(list(range(1, 129)), deadline)
        S = _local_search(S, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 5: Multiple random restarts with local search
    while time.time() < deadline:
        order = list(range(1, 129))
        rng.shuffle(order)
        S = _greedy_build(order, deadline)
        S = _local_search(S, deadline, max_iterations=10000)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    return list(best_set) if best_set else [1]
