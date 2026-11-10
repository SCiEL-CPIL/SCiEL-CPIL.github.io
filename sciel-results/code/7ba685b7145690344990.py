def construct():
    import time
    from math import gcd
    import random
    import sys

    deadline = time.time() + 82.0  # safety margin before 90s kill
    N = 128
    all_numbers = list(range(1, N + 1))

    # Precompute GCD table
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]

    # Precompute conflict structure for fast verification
    # For each pair (a,b), precompute the set of x such that gcd(a,b)==gcd(a,x)==gcd(b,x)
    # This allows O(1) check for whether adding x conflicts with pair (a,b)
    # But storing all pairs is 128*127/2 = 8128 pairs, each with up to 126 x values
    # Instead, we use a different approach: for each x, precompute which pairs (a,b) 
    # would form a bad triple with x.
    
    # For a given x, a pair (a,b) forms a bad triple with x iff:
    # gcd(a,b) == gcd(a,x) == gcd(b,x) = d
    # So d must divide x, and gcd(a/d, b/d) == 1, gcd(a/d, x/d) == 1, gcd(b/d, x/d) == 1
    # where d = gcd(a,x) = gcd(b,x)
    
    # Actually, let's think differently. For each x, and each possible gcd value d that divides x:
    # The "conflict partners" of x at level d are numbers a where gcd(a,x) = d.
    # A pair (a,b) both in this set conflicts with x iff gcd(a,b) = d.
    
    # Let's precompute: for each x, for each d|x, the list of a (a!=x) where gcd(a,x)=d
    # Then a bad triple involving x exists iff there are a,b in the same level-d list 
    # with gcd(a,b)=d.

    # Actually for the can_add check, we need: given current set S, can we add x?
    # This fails iff there exist a,b in S such that gcd(a,b)=gcd(a,x)=gcd(b,x).
    
    # Fast approach: for each x, precompute a "conflict graph" on the other numbers.
    # Two numbers a,b are "conflicting with respect to x" if gcd(a,b)==gcd(a,x)==gcd(b,x).
    # Then x can be added iff S contains no such conflicting pair.
    
    # Even faster: for each x, precompute the set of "bad pairs" P_x = {(a,b) : gcd(a,b)==gcd(a,x)==gcd(b,x)}
    # Then x can be added to S iff no pair in P_x is fully contained in S.
    
    # Number of bad pairs per x: at most C(127,2) ~ 8000, but typically much fewer.
    # Total storage: 128 * ~1000 pairs = ~128000 pairs, manageable.
    
    # Let's precompute bad_pairs[x] = list of (a,b) tuples where a<b, a!=x, b!=x,
    # and gcd(a,b)==gcd(a,x)==gcd(b,x)
    
    bad_pairs = [None] * (N + 1)
    for x in range(1, N + 1):
        bp = []
        for a in range(1, N + 1):
            if a == x:
                continue
            gax = g[a][x]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if g[a][b] == gax and g[b][x] == gax:
                    bp.append((a, b))
        bad_pairs[x] = bp
    
    # For even faster checking, precompute for each x a set of pairs (as frozensets or encoded ints)
    # Encode pair (a,b) as a * 129 + b (since b <= 128)
    bad_pair_sets = [None] * (N + 1)
    for x in range(1, N + 1):
        s = set()
        for a, b in bad_pairs[x]:
            s.add(a * 129 + b)
        bad_pair_sets[x] = s
    
    def can_add_fast(x, S_set):
        """Check if x can be added to S_set without creating a violating triple.
        O(|S_set|^2 / word_size) using precomputed bad pairs."""
        # Check if any bad pair of x is fully in S_set
        for a, b in bad_pairs[x]:
            if a in S_set and b in S_set:
                return False
        return True
    
    def can_add_fast_set(x, S_set):
        """Alternative: check using encoded pairs."""
        # Build set of all pairs in S_set
        # This is O(|S|^2) to build, so only useful if |S| is small
        # Better to just iterate bad_pairs[x]
        for a, b in bad_pairs[x]:
            if a in S_set and b in S_set:
                return False
        return True
    
    # Even faster: maintain a "pair presence" set for the current solution
    # pair_present = set of encoded pairs (a*129+b) for a<b both in S
    # Then can_add(x) = no pair in bad_pair_sets[x] is in pair_present
    # Adding x: add all pairs (x, y) for y in S to pair_present
    # Removing x: remove all pairs (x, y) for y in S from pair_present
    
    def solve_with_pair_maintenance():
        """Use pair-presence set for O(|bad_pairs[x]|) add checks."""
        best_S = []
        best_size = 0
        
        def try_greedy(order):
            nonlocal best_S, best_size
            S = set()
            pair_present = set()
            S_list = []
            for x in order:
                if x in S:
                    continue
                # Check if x can be added
                conflict = False
                for a, b in bad_pairs[x]:
                    if a in S and b in S:
                        conflict = True
                        break
                if not conflict:
                    S.add(x)
                    S_list.append(x)
                    # Add pairs involving x
                    for y in S_list[:-1]:
                        if y < x:
                            pair_present.add(y * 129 + x)
                        else:
                            pair_present.add(x * 129 + y)
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_S = list(S_list)
            return S_list
        
        def try_remove_add(S_list, time_budget):
            """Try removing one element and adding others."""
            nonlocal best_S, best_size
            S = set(S_list)
            S_list = list(S_list)
            improved = True
            while improved and time.time() < time_budget:
                improved = False
                for rem in S_list[:]:
                    if time.time() > time_budget:
                        break
                    # Remove rem
                    S_trial = S - {rem}
                    S_trial_list = [x for x in S_list if x != rem]
                    # Try to add candidates
                    candidates = [x for x in all_numbers if x not in S_trial]
                    added = 0
                    for x in candidates:
                        if time.time() > time_budget:
                            break
                        conflict = False
                        for a, b in bad_pairs[x]:
                            if a in S_trial and b in S_trial:
                                conflict = True
                                break
                        if not conflict:
                            S_trial.add(x)
                            S_trial_list.append(x)
                            added += 1
                    if len(S_trial) > len(S):
                        S = S_trial
                        S_list = list(S)
                        if len(S_list) > best_size:
                            best_size = len(S_list)
                            best_S = list(S_list)
                        improved = True
                        break
            return S_list
        
        def try_swap(S_list, time_budget):
            """Try removing one and adding one or more."""
            nonlocal best_S, best_size
            S = set(S_list)
            S_list = list(S_list)
            for rem in S_list[:]:
                if time.time() > time_budget:
                    break
                S_trial = S - {rem}
                S_trial_list = [x for x in S_list if x != rem]
                # Try adding multiple
                candidates = [x for x in all_numbers if x not in S_trial]
                for x in candidates:
                    if time.time() > time_budget:
                        break
                    conflict = False
                    for a, b in bad_pairs[x]:
                        if a in S_trial and b in S_trial:
                            conflict = True
                            break
                    if not conflict:
                        S_trial.add(x)
                        S_trial_list.append(x)
                if len(S_trial) > len(S):
                    S = S_trial
                    S_list = list(S)
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S = list(S_list)
                    break
            return S_list

        # Multiple greedy passes
        random.seed(42)
        
        # Various orderings
        orderings = []
        # Natural
        orderings.append(list(all_numbers))
        # Reversed
        orderings.append(list(reversed(all_numbers)))
        # By number of bad pairs (fewer first)
        orderings.append(sorted(all_numbers, key=lambda x: (len(bad_pairs[x]), x)))
        # By number of bad pairs (more first)
        orderings.append(sorted(all_numbers, key=lambda x: (-len(bad_pairs[x]), x)))
        # Random orderings
        for i in range(100):
            order = all_numbers[:]
            random.seed(i * 104729 + 37)
            random.shuffle(order)
            orderings.append(order)
        
        for order in orderings:
            if time.time() > deadline - 20:
                break
            S = try_greedy(order)
            if len(S) > best_size:
                best_size = len(S)
                best_S = list(S)
        
        # Local search on best
        if time.time() < deadline - 15:
            best_S = try_remove_add(best_S, deadline - 10)
        
        if time.time() < deadline - 10:
            best_S = try_swap(best_S, deadline - 5)
        
        # Iterated local search
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            # Perturb
            S_perturbed = set(best_S)
            k = min(3, len(S_perturbed))
            to_remove = random.sample(list(S_perturbed), k)
            for x in to_remove:
                S_perturbed.discard(x)
            S_perturbed_list = list(S_perturbed)
            
            # Re-optimize
            S_opt = try_remove_add(S_perturbed_list, deadline - 5)
            if time.time() < deadline - 3:
                S_opt = try_swap(S_opt, deadline - 3)
            if len(S_opt) > best_size:
                best_size = len(S_opt)
                best_S = list(S_opt)
        
        return best_S
    
    result = solve_with_pair_maintenance()
    
    # Final verification
    if not result:
        result = [1]
    
    # Verify no violations
    S_set = set(result)
    valid = True
    for i in range(len(result)):
        a = result[i]
        for j in range(i + 1, len(result)):
            b = result[j]
            gab = g[a][b]
            for k in range(j + 1, len(result)):
                c = result[k]
                if gab == g[a][c] == g[b][c]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Remove violating elements
        while not valid:
            violations = []
            for i in range(len(result)):
                a = result[i]
                for j in range(i + 1, len(result)):
                    b = result[j]
                    gab = g[a][b]
                    for k in range(j + 1, len(result)):
                        c = result[k]
                        if gab == g[a][c] == g[b][c]:
                            violations.append((a, b, c))
            if not violations:
                break
            # Remove element with most conflicts
            conf = {}
            for a, b, c in violations:
                conf[a] = conf.get(a, 0) + 1
                conf[b] = conf.get(b, 0) + 1
                conf[c] = conf.get(c, 0) + 1
            max_conf = -1
            to_remove = None
            for elem in result:
                cc = conf.get(elem, 0)
                if cc > max_conf:
                    max_conf = cc
                    to_remove = elem
            if to_remove is None:
                break
            result.remove(to_remove)
            valid = True
    
    if not result:
        result = [1]
    
    return result
