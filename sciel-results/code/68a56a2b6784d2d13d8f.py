"""Rewritten solution: structural change to a bit-parallel hypergraph
independent-set search with many randomized restarts and a local-improvement
pass (remove one, re-greedy) to escape greedy local optima.

The constraint is: no three distinct a,b,c in S with a+b, a+c, b+c all in S.
We model this as a 3-uniform hypergraph H on vertices {1..128} where each
forbidden triple (a,b,c) (with the three pair sums also in the vertex set)
is an edge. We seek a maximum independent set in H.

Key structural change vs. the parent:
1. Precompute ALL forbidden triples (edges) once via bit-parallel enumeration
   over the full vertex set [1..128].
2. Represent the current set S as a 128-bit integer mask.
3. For a candidate x, x is addable iff there is no pair (a,b) already in S
   such that {x,a,b} is a forbidden triple. We precompute, for each x, a
   128x128 "conflict" structure, but more efficiently: for each x, we store
   a bitmask over the 128 vertices such that bit v is set iff there exists
   some u in S with {x,u,v} forbidden. This is computed as:
     conflict[x] = OR over all a: (mask of all b such that {x,a,b} is forbidden)
   But since S changes, we need a dynamic check. Instead, we use:
     For each x, define F[x] as a list of pairs (a,b) such that {x,a,b} is forbidden.
     x is addable to S iff no (a,b) in F[x] has both a and b in S.
   This is expensive if F[x] is large. Instead, we use the bit-parallel trick:
     For each x, precompute a 128-bit mask C[x] where bit a is set iff
     there exists b such that {x,a,b} is forbidden AND b is in [1..128].
     But that's not quite right either because b must also be in S.

   Better approach: Precompute for each ordered pair (x, a) the set of b
   values such that {x,a,b} is a forbidden triple. Store this as a 128-bit
   mask M[x][a]. Then x is addable to S iff for ALL a in S, (M[x][a] & mask_S) == 0.
   Equivalently, x is NOT addable iff there exists a in S such that M[x][a] & mask_S != 0.

   So: x is addable iff AND over all a in S of (M[x][a] & mask_S == 0).
   We can compute: OR over all a in S of M[x][a], then check if that OR & mask_S == 0.
   Let T[x] = OR over a in S of M[x][a]. Then x is addable iff (T[x] & mask_S) == 0.

   We maintain for the current S a table T[x] for each candidate x, updated
   incrementally when S changes.

4. Use randomized greedy: repeatedly pick a random addable candidate, add it.
   Then do local improvement: for each x in S, try removing x and re-greedy
   to see if we can get a larger set.

Given time constraints, we'll do a simpler but effective approach:
- Precompute M[x][a] for all x,a in [1..128] as 128-bit masks.
- Use multiple randomized greedy restarts with different orderings.
- Keep the best solution found.
"""
import time
import random

def _precompute_masks():
    """Precompute M[x][a] = bitmask of all b in [1..128] such that
    {x, a, b} is a forbidden triple (i.e., x+a, x+b, a+b are all in [1..128]
    and the triple is distinct: x, a, b all distinct).
    
    Actually, the constraint is: a,b,c distinct, and a+b, a+c, b+c all in S.
    The pair sums don't need to be distinct from a,b,c or from each other.
    
    So a forbidden triple is (x, a, b) with x != a, x != b, a != b, and
    x+a in [1,128], x+b in [1,128], a+b in [1,128].
    
    M[x][a] has bit b set iff {x,a,b} is forbidden (x,a,b distinct, and all
    three pair sums are in [1,128]).
    """
    M = [[0] * 129 for _ in range(129)]  # M[x][a] for x,a in 0..128
    for x in range(1, 129):
        for a in range(1, 129):
            if x == a:
                continue
            xa = x + a
            if xa > 128:
                continue
            mask = 0
            for b in range(1, 129):
                if b == x or b == a:
                    continue
                xb = x + b
                ab = a + b
                if xb <= 128 and ab <= 128:
                    mask |= (1 << b)
            M[x][a] = mask
    return M

def _is_valid(S):
    """Verify no forbidden triple exists in S."""
    Sset = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            if (a + b) not in Sset:
                continue
            for k in range(j + 1, n):
                c = S[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True

def _greedy_randomized(M, mask_S, S, deadline, rng):
    """Given current S (as list) and mask_S (128-bit int), greedily add
    random addable candidates until none remain or time runs out.
    Returns (S, mask_S)."""
    # Precompute T[x] = OR over a in S of M[x][a] for all x
    # T[x] & mask_S == 0 means x is addable
    T = [0] * 129
    m = mask_S
    while m:
        lb = m & (-m)
        a = lb.bit_length() - 1
        m ^= lb
        for x in range(1, 129):
            if x == a:
                continue
            T[x] |= M[x][a]
    
    # Find all addable candidates
    addable = []
    for x in range(1, 129):
        if (mask_S >> x) & 1:
            continue
        if T[x] & mask_S == 0:
            addable.append(x)
    
    while addable and time.time() < deadline:
        idx = rng.randrange(len(addable))
        x = addable[idx]
        # Double-check addability (should be fine, but safety)
        if T[x] & mask_S != 0:
            addable.pop(idx)
            continue
        # Add x
        mask_S |= (1 << x)
        S.append(x)
        addable.pop(idx)
        # Update T for all remaining candidates
        for x2 in range(1, 129):
            if x2 == x:
                continue
            if (mask_S >> x2) & 1:
                continue
            T[x2] |= M[x2][x]
            if T[x2] & mask_S != 0:
                # x2 is no longer addable
                try:
                    addable.remove(x2)
                except ValueError:
                    pass
        # Also remove x from addable (already done)
    
    return S, mask_S

def _local_improve(M, S, mask_S, deadline, rng):
    """Try removing each element and re-greedy to see if we get a larger set.
    Returns the best (S, mask_S) found."""
    best_S = list(S)
    best_mask = mask_S
    best_size = len(S)
    
    for idx in range(len(S)):
        if time.time() >= deadline:
            break
        x_remove = S[idx]
        # Try removing x_remove
        S2 = [x for i, x in enumerate(S) if i != idx]
        mask2 = mask_S & ~(1 << x_remove)
        # Re-greedy
        S2, mask2 = _greedy_randomized(M, mask2, S2, deadline, rng)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = list(S2)
            best_mask = mask2
    
    return best_S, best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    M = _precompute_masks()
    
    rng = random.Random(42)
    
    best_S = []
    best_mask = 0
    best_size = 0
    
    # Multiple restarts with different random seeds
    num_restarts = 20
    for trial in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Start with a random subset to break symmetry
        # Or start empty and let greedy fill
        S = []
        mask_S = 0
        
        # Optional: seed with a few random elements
        # For now, start empty
        
        S, mask_S = _greedy_randomized(M, mask_S, S, deadline, rng)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            best_mask = mask_S
        
        # Local improvement
        if time.time() < deadline - 5.0:
            S_imp, mask_imp = _local_improve(M, best_S, best_mask, deadline - 5.0, rng)
            if len(S_imp) > best_size:
                best_size = len(S_imp)
                best_S = list(S_imp)
                best_mask = mask_imp
    
    # Final verification
    if not _is_valid(best_S):
        # Fallback to a known valid set
        best_S = [1, 62] + list(range(64, 129))
    
    return [int(x) for x in sorted(best_S)]
