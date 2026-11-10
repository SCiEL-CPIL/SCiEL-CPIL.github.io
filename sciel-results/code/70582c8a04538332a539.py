"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_forbidden(N=180):
    """
    Precompute for each a in [1, N], the set of pairs (b, c) with b < c
    such that 1/a = 1/b + 1/c, i.e., (b-a)(c-a) = a^2.
    b, c must be in [1, N] and distinct from a and each other.
    """
    forbidden_pairs = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        # d = b-a, so c-a = a2/d. b = a+d, c = a+a2/d.
        # d must be a divisor of a2.
        # b, c in [1, N], b != c, b != a, c != a.
        for d in range(1, int(a2 ** 0.5) + 1):
            if a2 % d != 0:
                continue
            for dd in (d, a2 // d):
                if dd == 0:
                    continue
                # Case 1: d > 0
                b1 = a + dd
                c1 = a + a2 // dd
                if b1 == c1: continue
                if 1 <= b1 <= N and 1 <= c1 <= N and b1 != a and c1 != a:
                    if b1 < c1:
                        forbidden_pairs[a].add((b1, c1))
                    else:
                        forbidden_pairs[a].add((c1, b1))
                
                # Case 2: d < 0 (i.e., dd is positive, but we use -dd)
                b2 = a - dd
                c2 = a - a2 // dd
                if b2 == c2: continue
                if 1 <= b2 <= N and 1 <= c2 <= N and b2 != a and c2 != a:
                    if b2 < c2:
                        forbidden_pairs[a].add((b2, c2))
                    else:
                        forbidden_pairs[a].add((c2, b2))
    
    return forbidden_pairs

def _can_add(S_set, elem, forbidden_pairs):
    """
    Check if adding `elem` to `S_set` (a set) creates any forbidden triple.
    We only need to check triples involving `elem`.
    """
    # 1. Check if `elem` is the 'a' in 1/a = 1/b + 1/c
    for (b, c) in forbidden_pairs[elem]:
        if b in S_set and c in S_set:
            return False
    
    # 2. Check if `elem` is the 'b' or 'c' in 1/a = 1/b + 1/c
    # This is equivalent to checking for each a in S_set, if there is a pair (b,c) in forbidden_pairs[a]
    # such that one is `elem` and the other is in S_set.
    for a in S_set:
        for (b, c) in forbidden_pairs[a]:
            if b == elem and c in S_set:
                return False
            if c == elem and b in S_set:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    # Precompute forbidden pairs
    forbidden_pairs = _precompute_forbidden(N)
    
    # Strategy:
    # 1. Start with U = {91, ..., 180}. This set is valid.
    # 2. Try to add odd numbers from 1 to 90.
    # 3. Use local search to improve.
    
    S = set(range(91, N + 1))
    
    # Candidates to add: odd numbers from 1 to 90
    # We iterate in descending order? Or ascending?
    # Larger a in lower half might have more conflicts?
    # Let's try descending to see if we can keep the "larger" lower-half odds.
    candidates = list(range(89, 0, -2))
    
    # Greedy addition
    for a in candidates:
        if time.time() > deadline - 10:
            break
        if _can_add(S, a, forbidden_pairs):
            S.add(a)
    
    # Local Search / Hill Climbing
    # Try to add any remaining element (even numbers 1-90, or if we skipped some odds)
    # And try swaps.
    
    all_elements = set(range(1, N + 1))
    rng = random.Random(42)
    
    # Phase 2: Try to add any missing element
    missing = list(all_elements - S)
    rng.shuffle(missing)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for elem in missing:
            if time.time() > deadline - 5:
                break
            if _can_add(S, elem, forbidden_pairs):
                S.add(elem)
                missing.remove(elem)
                improved = True
    
    # Phase 3: Try swaps to increase size (unlikely to increase size if we just added all possible,
    # but maybe we can swap a large element for two smaller ones? No, swap is 1-for-1.
    # So size only increases by additions.
    # However, we might have blocked an addition by a bad element.
    # Try removing one element and adding two? That's complex.
    # Let's stick to single additions which we already did.
    # Maybe try a few random restarts if time permits?
    # Given the strong baseline of U + odds, restarts might not help much.
    
    return list(S)

# EVOLVE-BLOCK-END
