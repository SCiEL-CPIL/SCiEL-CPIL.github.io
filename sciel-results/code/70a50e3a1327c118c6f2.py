"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

DEADLINE = time.time() + 80  # safety margin before 90s kill

def _precompute():
    N = 128
    # Precompute GCD matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]
    return G, N

def _find_bad_triples(G, N):
    """Find all triples (a,b,c) with a<b<c such that gcd(a,b)==gcd(a,c)==gcd(b,c)."""
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = G[a][b]
            for c in range(b + 1, N + 1):
                if G[a][c] == gab and G[b][c] == gab:
                    bad_triples.append((a, b, c))
    return bad_triples

def _greedy_hitting_set(bad_triples, N):
    """
    Greedy hitting set: remove elements to break all bad triples.
    Returns the remaining set S.
    """
    S = set(range(1, N + 1))
    
    # For each element, list of triples it's in
    triples_by_elem = {i: [] for i in range(1, N + 1)}
    for t in bad_triples:
        for x in t:
            triples_by_elem[x].append(t)
    
    # Active triples: those where all 3 elements are in S
    active_triples = set(bad_triples)
    
    # Count active triples per element
    active_count = {i: 0 for i in range(1, N + 1)}
    for t in active_triples:
        for x in t:
            active_count[x] += 1
    
    while active_triples:
        if time.time() > DEADLINE:
            break
        
        # Find element with max active_count
        best_elem = -1
        max_count = -1
        for x in S:
            if active_count[x] > max_count:
                max_count = active_count[x]
                best_elem = x
        
        if best_elem == -1 or max_count == 0:
            # Pick any element from first active triple
            t = next(iter(active_triples))
            for x in t:
                if x in S:
                    best_elem = x
                    break
            if best_elem == -1:
                break
        
        S.remove(best_elem)
        
        # Update: all triples containing best_elem are no longer active
        for t in triples_by_elem[best_elem]:
            if t in active_triples:
                active_triples.remove(t)
                for x in t:
                    if x != best_elem:
                        active_count[x] -= 1
    
    return S

def _local_search(S, G, N, bad_triples):
    """
    Try to improve S by adding back removed elements or swapping.
    Returns improved S.
    """
    S = set(S)
    removed = set(range(1, N + 1)) - S
    
    # Precompute for each element x not in S, the set of pairs (a,b) in S 
    # such that (a,b,x) would be a bad triple.
    # If we add x, we need no such pair (a,b) both in S.
    
    # For each x, compute conflict_pairs[x] = set of pairs (a,b) in S where 
    # {a,b,x} is a bad triple.
    # Actually, we need: for x to be addable, there should be NO pair (a,b) in S
    # such that gcd(a,b)==gcd(a,x)==gcd(b,x).
    
    # Let's precompute for each x, the list of elements y such that there exists
    # a "witness" pair. Actually simpler: for each x, check if adding x creates
    # any bad triple with two existing elements in S.
    
    # For efficiency, for each x, we can check all pairs (a,b) in S.
    # But that's O(|S|^2) per x, which is too slow.
    
    # Better: for each x, precompute the set of "conflicting pairs" in S.
    # A pair (a,b) in S conflicts with x if gcd(a,b)==gcd(a,x)==gcd(b,x).
    
    # Let's precompute for each x, a set of "forbidden pairs" in S.
    # Actually, let's just try to add each removed element and check validity.
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for x in sorted(removed, key=lambda v: -v):  # Try larger numbers first?
            # Check if x can be added to S
            can_add = True
            for a in S:
                if a >= x:
                    break
                ga = G[x][a]
                for b in S:
                    if b <= a:
                        continue
                    if b >= x:
                        break
                    if G[x][b] == ga and G[a][b] == ga:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                S.add(x)
                removed.remove(x)
                improved = True
    
    return S

def _swap_search(S, G, N, bad_triples):
    """
    Try 1-for-1 and 2-for-1 swaps to improve S.
    """
    S = set(S)
    removed = set(range(1, N + 1)) - S
    
    def is_valid_set(S):
        """Check if S has no bad triple."""
        for a in S:
            for b in S:
                if a >= b:
                    continue
                gab = G[a][b]
                for c in S:
                    if c <= b:
                        continue
                    if G[a][c] == gab and G[b][c] == gab:
                        return False
        return True
    
    # Try adding one element and removing one or two elements
    for x in sorted(removed):
        if time.time() > DEADLINE:
            break
        # Check if x can be added if we remove some elements
        # Find all elements in S that conflict with x
        # An element a in S conflicts with x if there exists b in S such that
        # gcd(a,b)==gcd(a,x)==gcd(b,x).
        # Actually, x conflicts with pair (a,b) in S if gcd(a,b)==gcd(a,x)==gcd(b,x).
        
        # Find all pairs (a,b) in S that would form bad triple with x
        conflicting_pairs = []
        for a in S:
            if a >= x:
                continue
            ga = G[x][a]
            for b in S:
                if b <= a:
                    continue
                if b >= x:
                    break
                if G[x][b] == ga and G[a][b] == ga:
                    conflicting_pairs.append((a, b))
        
        if not conflicting_pairs:
            # Can add x directly
            S.add(x)
            removed.remove(x)
            continue
        
        # Try removing 1 element to break all conflicting pairs
        # Find element that appears in most conflicting pairs
        from collections import Counter
        elem_count = Counter()
        for a, b in conflicting_pairs:
            elem_count[a] += 1
            elem_count[b] += 1
        
        # Try removing the most frequent element
        if elem_count:
            best_remove = max(elem_count, key=lambda k: elem_count[k])
            S_new = S - {best_remove}
            S_new.add(x)
            # Verify no other bad triples created
            if is_valid_set(S_new):
                S = S_new
                removed.remove(x)
                removed.add(best_remove)
                continue
        
        # Try removing 2 elements
        if len(conflicting_pairs) <= 10:  # Limit search
            # Try all pairs of elements to remove
            candidates = set()
            for a, b in conflicting_pairs:
                candidates.add(a)
                candidates.add(b)
            candidates = list(candidates)
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    S_new = S - {candidates[i], candidates[j]}
                    S_new.add(x)
                    if is_valid_set(S_new):
                        S = S_new
                        removed.remove(x)
                        removed.add(candidates[i])
                        removed.add(candidates[j])
                        break
                if x not in removed:
                    break
    
    return S

def construct():
    G, N = _precompute()
    bad_triples = _find_bad_triples(G, N)
    
    # Phase 1: Greedy hitting set
    S = _greedy_hitting_set(bad_triples, N)
    
    # Phase 2: Local search - add back elements
    S = _local_search(S, G, N, bad_triples)
    
    # Phase 3: Swap search
    S = _swap_search(S, G, N, bad_triples)
    
    # Phase 4: Another round of local search
    S = _local_search(S, G, N, bad_triples)
    
    if not S:
        return [1]
        
    return sorted(S)

# EVOLVE-BLOCK-END
