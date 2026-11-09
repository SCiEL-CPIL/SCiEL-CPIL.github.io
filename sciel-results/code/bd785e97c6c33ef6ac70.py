"""Rewritten program: parity seed + greedy/local-search to add even elements."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        # S is a set; check no a+b=c with a,b,c in S (a=b allowed)
        L = list(S)
        n = len(L)
        # For each pair (i,j) with i<=j, check if L[i]+L[j] in S
        # Use set for O(1) lookup
        for i in range(n):
            a = L[i]
            for j in range(i, n):
                b = L[j]
                if (a + b) in S:
                    return False
        return True

    def solve_one(vals):
        # Start with all odd elements (guaranteed sum-free)
        odds = [x for x in vals if x % 2 != 0]
        evens = [x for x in vals if x % 2 == 0]
        S = set(odds)
        # Greedy: try adding evens in order of "safety" (fewest conflicts)
        # For an even e, it conflicts if:
        #   (1) exists o1,o2 in S with o1+o2 = e  [both odd -> even]
        #   (2) exists o in S, e2 in S(evens added) with o+e2 = e or e2+o = e
        #   (3) exists e1,e2 in S(evens added) with e1+e2 = e or e+e2 = e (e=e2 means 2e)
        # We track S as a set; for efficiency, precompute.
        # Sort evens by absolute value (smaller evens more likely to be sums of two odds)
        # Actually, larger magnitude evens are less likely to be o1+o2 since odds in [-10000,10000]
        # Let's just try all evens, pick those that don't conflict, greedily.
        # To maximize size, try evens in random order? Or by conflict count.
        # Compute conflict count for each even with current S (only odds initially)
        def conflict_with_e(e, S):
            # Check if adding e to S creates any a+b=c
            # New triples involving e:
            #   e + x = y  => y - x = e for x,y in S
            #   x + e = y  => y - x = e for x,y in S  (same as above)
            #   x + y = e  => x + y = e for x,y in S
            #   e + e = y  => y = 2e in S
            #   x + e = e  => x = 0, but 0 not in vals (nonzero)
            # So check:
            #   (a) exists x,y in S with x+y = e
            #   (b) exists x,y in S with y - x = e  (i.e., x+e=y)
            #   (c) 2e in S
            S_list = list(S)
            # (a) x+y=e: for x in S, check e-x in S
            for x in S_list:
                if (e - x) in S:
                    # x + (e-x) = e, both in S -> conflict
                    return True
            # (b) x+e=y: for x in S, check x+e in S
            for x in S_list:
                if (x + e) in S:
                    return True
            # (c) 2e in S
            if (2 * e) in S:
                return True
            return False

        # Greedy add evens: sort by how "easy" they are
        # Try multiple orderings? Keep it simple: sort by abs value descending (larger first, fewer conflicts)
        evens_sorted = sorted(evens, key=lambda x: -abs(x))
        for e in evens_sorted:
            if time.time() > DEADLINE:
                break
            if not conflict_with_e(e, S):
                S.add(e)
        # Local search: try to add more by removing conflicting elements? 
        # Or try different orderings of evens.
        # Try a few random orderings to see if we get larger sets
        best = S
        from random import Random
        rng = Random(42)
        for trial in range(50):
            if time.time() > DEADLINE:
                break
            trial_S = set(odds)
            evens_shuffled = evens[:]
            rng.shuffle(evens_shuffled)
            for e in evens_shuffled:
                if not conflict_with_e(e, trial_S):
                    trial_S.add(e)
            if len(trial_S) > len(best):
                best = trial_S
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            # Fallback: just odds
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        vals = instance(size, seed)
        results.append(solve_one(vals))
    return results
# EVOLVE-BLOCK-END
