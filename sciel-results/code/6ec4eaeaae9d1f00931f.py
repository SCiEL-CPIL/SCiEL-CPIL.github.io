import time
import random
import itertools

N = 128
DEADLINE = 85.0  # safety margin under 90s

def _check_valid(Sset):
    """Full O(n^3) validity check. Returns True if valid."""
    lst = sorted(Sset)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            ab = a + b
            if ab not in Sset:
                continue
            for k in range(j+1, n):
                c = lst[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True

def _count_violations(Sset):
    """Count the number of forbidden triples (a,b,c) with all pair sums in S."""
    lst = sorted(Sset)
    n = len(lst)
    total = 0
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            ab = a + b
            if ab not in Sset:
                continue
            for k in range(j+1, n):
                c = lst[k]
                if (a + c) in Sset and (b + c) in Sset:
                    total += 1
    return total

def _can_add(Sset, x):
    """Check if adding x to Sset creates any violation."""
    if x in Sset:
        return False
    lst = sorted(Sset)
    for i in range(len(lst)):
        a = lst[i]
        if (x + a) not in Sset:
            continue
        for j in range(i+1, len(lst)):
            b = lst[j]
            if (x + b) in Sset and (a + b) in Sset:
                return False
    return True

def _greedy_construct():
    """Greedy construction: add elements in order of minimal conflict."""
    Sset = set()
    # Try adding elements from 1 to N, prefer those that cause least future conflict
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(Sset, x):
            Sset.add(x)
    return Sset

def _local_search(Sset, max_iters=2000):
    """Local search: try removing one element and adding another to improve size."""
    Sset = set(Sset)
    best_size = len(Sset)
    rng = random.Random(42)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Try a random swap: remove y, add x
        notin = [x for x in range(1, N+1) if x not in Sset]
        if not notin:
            break
        
        x = rng.choice(notin)
        y = rng.choice(list(Sset))
        
        # Remove y
        S2 = Sset - {y}
        
        # Try to add x
        if _can_add(S2, x):
            S2.add(x)
            # Try to add back more elements
            improved = True
            while improved:
                improved = False
                for z in range(1, N+1):
                    if z not in S2 and _can_add(S2, z):
                        S2.add(z)
                        improved = True
            
            if len(S2) > best_size:
                best_size = len(S2)
                Sset = S2
    
    return Sset

def _two_part_construct():
    """Two-part construction: L (weakly sum-free) union U (sum-free).
    Choose gap so that 2*max(L) < min(U) to prevent mixed-sum violations."""
    best_set = set()
    best_size = 0
    
    # Try different split points
    for split in range(10, 80):
        # L is from 1..split, U is from split+1..128
        # For U to be sum-free within itself, use upper half
        # For L to be weakly sum-free, use odd numbers or similar
        
        # Strategy: L = odd numbers in 1..split, U = numbers in (split, 128] that are sum-free
        L = set(x for x in range(1, split+1) if x % 2 == 1)  # odd numbers are sum-free
        
        # U: pick a sum-free subset of split+1..128
        # Simplest: take numbers in (split, 128] that are > split/2... but they're all > split
        # Actually, any subset of (split, 128] where all elements > split is sum-free
        # because sum of two elements > 2*split >= split+1, but we need sum NOT in U
        # If U = {split+1, ..., 128}, then a+b for a,b in U is >= 2*(split+1)
        # We need 2*(split+1) > 128, i.e., split > 63
        # Otherwise, some sums land in U
        
        # Better: make U = {u_min, ..., 128] where u_min > 128/2 = 64
        # Then any sum of two U elements > 128, so not in 1..128, hence not in U
        
        # So: L = odd numbers in 1..split, U = all numbers in 65..128
        # But we need 2*max(L) < min(U) = 65
        # max(L) <= split, so 2*split < 65, i.e., split <= 32
        
        if split <= 32:
            L = set(x for x in range(1, split+1) if x % 2 == 1)
            U = set(range(65, 129))
            Sset = L | U
            
            # Verify validity
            if _check_valid(Sset):
                if len(Sset) > best_size:
                    best_size = len(Sset)
                    best_set = set(Sset)
        
        # Also try: L = {1..split} \ {some}, U = {65..128}
        # Or different gap strategies
        
    # Try: L = {1,3,5,...,2k-1} (odd up to 2k-1), U = {m, m+1, ..., 128}
    # Need: 2*(2k-1) < m, and m > 64 for U to be sum-free
    # So 4k-2 < m and m >= 65
    # Maximize |L| + |U| = k + (128-m+1) = k + 129-m
    # With m > 4k-2, so m >= 4k-1
    # Size = k + 129 - (4k-1) = 130 - 3k
    # This decreases with k, so k=1 gives 127, but m>=3, U={3..128} size 126, L={1} size 1, total 127
    # But wait, U={3..128} is not sum-free: 3+3=6 in U. We need distinct a,b,c.
    # Actually the constraint is on distinct a,b,c. Let me re-check.
    
    # The constraint: no three DISTINCT members a,b,c such that a+b, a+c, b+c are all in S.
    # So for U to be "safe", we need that for any distinct a,b,c in U, at least one of a+b, a+c, b+c is not in U.
    # If U = {m, ..., 128], then a+b >= 2m. If 2m > 128, i.e., m > 64, then a+b > 128, so a+b not in S (since S subset of 1..128).
    # So U = {65..128} works: any sum of two distinct elements >= 65+66=131 > 128.
    
    # For L: we need that for any distinct a,b,c in L, at least one pair sum is not in S.
    # If L is sum-free (no a+b in L for distinct a,b in L), then pair sums within L are not in L.
    # But pair sums could be in U! We need 2*max(L) < min(U) to ensure pair sums of L are < min(U).
    
    # So: L sum-free, 2*max(L) < 65, i.e., max(L) <= 32.
    # Best sum-free subset of 1..32: odd numbers {1,3,...,31}, size 16.
    # U = {65..128}, size 64.
    # Total: 16 + 64 = 80.
    
    # Can we do better? What if L is not fully sum-free but has no "forbidden triple"?
    # A forbidden triple in L would be distinct a,b,c in L with a+b, a+c, b+c all in S.
    # If a+b, a+c, b+c are all in U (since 2*max(L) < min(U) is violated), that's bad.
    # But if some pair sums are in L, that's also bad if all three pair sums are in S.
    
    # Let's just use the clean construction: L=odd(1..32), U={65..128}, size 80.
    # Then try local search to improve.
    
    L = set(x for x in range(1, 33) if x % 2 == 1)
    U = set(range(65, 129))
    Sset = L | U
    
    if not _check_valid(Sset):
        # Fallback
        Sset = set(range(65, 129))
    
    # Try local search
    Sset = _local_search(Sset, max_iters=3000)
    
    return Sset

def _random_restart_construct():
    """Random restart with greedy + local search."""
    best_set = set()
    best_size = 0
    rng = random.Random(7)
    
    for trial in range(20):
        if time.time() > DEADLINE - 10:
            break
        
        Sset = _greedy_construct()
        Sset = _local_search(Sset, max_iters=500)
        
        if len(Sset) > best_size:
            best_size = len(Sset)
            best_set = set(Sset)
    
    return best_set

def construct():
    start = time.time()
    
    # Start with the two-part construction (known to give ~80)
    best_set = _two_part_construct()
    best_size = len(best_set)
    
    # Try random restarts to see if we can do better
    if time.time() < DEADLINE - 15:
        rs_set = _random_restart_construct()
        if len(rs_set) > best_size:
            best_set = rs_set
            best_size = len(rs_set)
    
    # Final local search on the best set
    if time.time() < DEADLINE - 5:
        best_set = _local_search(best_set, max_iters=5000)
    
    # Verify validity
    if not _check_valid(best_set):
        # Fallback to safe set
        best_set = set(range(65, 129))
    
    res = sorted(best_set)
    return [int(x) for x in res]
