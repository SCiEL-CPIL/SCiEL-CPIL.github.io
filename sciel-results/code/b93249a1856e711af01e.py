"""Combined sum-free subset search: odd baseline + efficient local search."""
# EVOLVE-BLOCK-START
import time

def construct():
    t_start = time.time()
    DEADLINE = 135.0  # safety margin under 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def local_search(S0, S_init, deadline):
        """Local search starting from S_init. S0 is the full candidate set.
        Maintain: S (set), blocked (set of elements in S0 that cannot be added),
        blockers (dict: y -> set of x in S that block y).
        """
        S = set(S_init)
        # Build blocked and blockers
        blocked = set()
        blockers = {}  # y -> set of x in S that block y
        for x in S:
            # x blocks 2x (if 2x in S0) and x/2 (if x even and x/2 in S0)
            for y in (2 * x,):
                if y in S0:
                    blocked.add(y)
                    blockers.setdefault(y, set()).add(x)
            if x % 2 == 0:
                y = x // 2
                if y in S0:
                    blocked.add(y)
                    blockers.setdefault(y, set()).add(x)
        # Note: elements in S are not "blocked" in the sense of being addable;
        # we only track blocked for elements NOT in S.
        # Actually, blocked should track elements in S0 \ S that cannot be added.
        # Let me redefine: blocked = {y in S0 \ S : y cannot be added to S}
        # But if y in S, it's not a candidate. Let's keep blocked as a set of
        # elements in S0 that are "conflicting" with S (cannot be added).
        # Elements in S are trivially not addable. So blocked is for S0 \ S.
        
        # Recompute blocked properly: y in S0 \ S is blocked if:
        #   2y in S, or (y even and y/2 in S), or exists a in S such that a + (y-a) ... 
        # Wait, the condition for y to be addable to S: no a, b in S with a+b=y, and 2y not in S.
        # So y is blocked if: (2y in S) OR (exists a in S: (y-a) in S).
        # The "exists a in S: (y-a) in S" is equivalent to: y is a sum of two elements of S.
        # This is hard to maintain incrementally. Let me use a different approach.
        
        # Alternative: maintain for each y in S0 \ S, the count of "blocking pairs" in S.
        # A pair (a,b) in S blocks y if a+b=y. Also 2y in S blocks y (since a=b=y... no, 2y in S means y+y=2y, so y is blocked if 2y in S? No: the condition is no a,b in S with a+b=y. If 2y in S, that doesn't directly block y. Wait: a+b=y with a,b in S. If y is in S, then we'd need a+b=y with a,b in S. But y itself is in S, so a=y, b=0? No, 0 not in S. Hmm.
        # Let me re-read: "no a + b = c with a, b, c in it (a = b allowed)".
        # So if y is in the set, we need: no a, b in set with a+b=y. And also no a,b in set with a+b=2y? No, that's not required. The condition is: for all a, b, c in set, a+b != c.
        # So y can be added to S iff: (no a, b in S with a+b=y) AND (for all a in S, a+y != ... wait, we also need that adding y doesn't create a+b=c where c=y, i.e., a+b=y, and where c is some existing element, i.e., a+y=c or y+a=c for some a,c in S, and where c=2y, i.e., y+y=2y so if 2y in S, then y+y=2y is a violation.
        # So y can be added iff:
        #   1. No a, b in S with a+b = y
        #   2. No a in S with a + y in S  (i.e., y + a = c for some a, c in S)
        #   3. 2y not in S
        # Condition 2 is equivalent to: no a in S such that (a+y) in S.
        
        # This is getting complex. Let me just do a simpler but still efficient approach:
        # For each candidate y not in S, check if it can be added by iterating over S.
        # With |S| <= 240, this is O(240) per candidate, and |S0| <= 240, so O(240*240) = O(57600) per pass.
        # With 400 passes, that's ~23M ops, which is fast in Python? Maybe not. Let me optimize.
        
        # Better: precompute for each y in S0, the set of a in S such that (y-a) in S.
        # But S changes. Let me just do the straightforward check but with early exit.
        
        def can_add(y, S):
            if y in S:
                return False
            # Check 2y not in S
            if 2 * y in S:
                return False
            # Check no a, b in S with a+b=y
            for a in S:
                if (y - a) in S:
                    return False
            # Check no a in S with a+y in S
            for a in S:
                if (a + y) in S:
                    return False
            return True

        def evaluate_remove(x, S):
            """If we remove x from S, how many new elements can we add?"""
            S2 = S - {x}
            gain = 0
            for y in S0:
                if y in S2:
                    continue
                if can_add(y, S2):
                    # Also check it wasn't already addable to S
                    if not can_add(y, S):
                        gain += 1
            return gain

        # Main loop
        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            # Try adding
            for y in S0:
                if y in S:
                    continue
                if can_add(y, S):
                    S.add(y)
                    improved = True
            # Try removing (only if it enables net gain)
            for x in list(S):
                if time.time() > deadline:
                    break
                gain = evaluate_remove(x, S)
                if gain > 1:
                    S.discard(x)
                    improved = True
            if not improved:
                break
        return S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S0 = instance(size, seed)
        S0_set = set(S0)
        
        # Starting points
        odd = [x for x in S0 if x % 2]
        top = S0[len(S0)//2:]
        bottom = S0[:len(S0)//2]
        
        best = None
        best_len = -1
        
        for init in [odd, top, bottom]:
            if time.time() > DEADLINE - 10:
                break
            cand = local_search(S0_set, init, DEADLINE - 5)
            if len(cand) > best_len:
                best_len = len(cand)
                best = cand
        
        # Also try greedy from empty
        if time.time() > DEADLINE - 15:
            if best is None:
                best = set(odd)
        else:
            empty_init = []
            cand = local_search(S0_set, empty_init, DEADLINE - 5)
            if len(cand) > best_len:
                best_len = len(cand)
                best = cand
        
        if best is None:
            best = set(odd)
        results.append(sorted(best))
    
    return results
# EVOLVE-BLOCK-END
