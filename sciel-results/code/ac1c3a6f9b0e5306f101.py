import time

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill

    # Precompute conflict graph: edge (i,j) if there exists k != i,j with 1/i=1/j+1/k or 1/j=1/i+1/k or 1/k=1/i+1/j
    # For three distinct a,b,c: 1/a = 1/b + 1/c means a = bc/(b+c)
    # So for pair (i,j), they conflict if there exists k such that {i,j,k} is a forbidden triple.
    # The forbidden triple condition: one element equals the "harmonic mean partner" of the other two.
    # Specifically, for distinct i,j,k: 1/i = 1/j + 1/k iff i = j*k/(j+k)
    # So (i,j) conflict if: exists k != i,j in [1,180] such that j*k/(j+k) = i (integer)
    #   OR i*k/(i+k) = j (integer)
    #   OR i*j/(i+j) = k (integer)
    # Actually, the triple is forbidden if ANY of the three elements equals the harmonic-mean-partner of the other two.
    # For a set S, it's valid iff no three distinct elements a,b,c in S satisfy 1/a=1/b+1/c.
    # This means: for any pair (b,c) in S, the value a=bc/(b+c) (if integer and in [1,180]) must NOT be in S (and must be distinct from b,c).
    
    # Precompute: for each pair (i,j), does adding both create a conflict?
    # Actually, the conflict is a 3-way constraint, not 2-way. So we can't simply use a 2-uniform graph.
    # Instead, we use a greedy + local search approach with efficient safety checks.

    # Precompute for each x in [1,180]: the set of pairs (b,c) such that x = b*c/(b+c), i.e., b*c = x*(b+c)
    # b*c - x*b - x*c = 0 => (b-x)*(c-x) = x^2
    # So for a given x, the pairs (b,c) with b*c/(b+c)=x are those where (b-x)*(c-x)=x^2.
    
    # For efficiency: precompute conflicts[x] = set of pairs that would conflict with x
    # But actually, we need: given a set S, is it safe to add x? 
    # x is unsafe if there exist b,c in S, b!=c, b!=x, c!=x, such that x = b*c/(b+c)
    # Equivalently: (b-x)*(c-x) = x^2 with b,c in S.
    
    # Also, x could be one of the "b" or "c" roles: 1/b = 1/x + 1/c means b = x*c/(x+c)
    # So b = x*c/(x+c) => (b-x)*(c-x) = x^2 as well! Same equation.
    # And 1/c = 1/x + 1/b => c = x*b/(x+b) => same equation.
    
    # So the key insight: for a given x, the forbidden pairs (b,c) are exactly those 
    # where (b-x)*(c-x) = x^2, with b != c, b != x, c != x.
    
    # Precompute for each x: list of (b,c) pairs in [1,180] that conflict with x
    conflict_pairs = {}
    for x in range(1, 181):
        pairs = []
        x2 = x * x
        # (b-x)*(c-x) = x^2, with b,c in [1,180], b != c, b != x, c != x
        # Let u = b-x, v = c-x, then u*v = x^2
        # b = x+u, c = x+v
        # b in [1,180] => 1 <= x+u <= 180 => 1-x <= u <= 180-x
        # Similarly for v
        # u and v are integers with u*v = x^2, u != 0, v != 0
        # Also b != c means u != v, and b != x means u != 0 (already), c != x means v != 0 (already)
        
        # Find all divisors of x^2
        for u in range(1, x2 + 1):
            if x2 % u != 0:
                continue
            v = x2 // u
            b = x + u
            c = x + v
            if 1 <= b <= 180 and 1 <= c <= 180 and b != c:
                pairs.append((b, c))
            # Also try negative u
            u_neg = -u
            b_neg = x + u_neg
            c_pos = x + v
            if 1 <= b_neg <= 180 and 1 <= c_pos <= 180 and b_neg != c_pos:
                pairs.append((b_neg, c_pos))
            # u positive, v negative
            b_pos = x + u
            c_neg = x - v
            if 1 <= b_pos <= 180 and 1 <= c_neg <= 180 and b_pos != c_neg:
                pairs.append((b_pos, c_neg))
            # Both negative
            b_nn = x - u
            c_nn = x - v
            if 1 <= b_nn <= 180 and 1 <= c_nn <= 180 and b_nn != c_nn:
                pairs.append((b_nn, c_nn))
        
        # Remove duplicates
        conflict_pairs[x] = set(pairs)
    
    # Greedy construction: start with all odds, try to add evens
    result = set(i for i in range(1, 181) if i % 2 == 1)
    
    # Try to add each even number
    evens = list(range(2, 181, 2))
    
    def is_safe(x, S):
        """Check if x can be added to set S without creating a forbidden triple."""
        # x is unsafe if there exist b,c in S such that (b-x)*(c-x) = x^2
        for (b, c) in conflict_pairs[x]:
            if b in S and c in S:
                return False
        return True
    
    for x in evens:
        if time.time() > deadline:
            break
        if is_safe(x, result):
            result.add(x)
    
    # Local search: 1-2 exchange moves (remove 1, add 2) to escape local optima
    all_nums = set(range(1, 181))
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        # Try to add a single element
        for x in sorted(all_nums - result):
            if time.time() > deadline:
                break
            if is_safe(x, result):
                result.add(x)
                improved = True
                break  # restart scan after one add
        
        if not improved:
            # Try 1-2 exchange: remove one element, add two
            for rem in list(result):
                if time.time() > deadline:
                    break
                S_minus = result - {rem}
                # Try adding pairs from all_nums - result
                candidates = list(all_nums - result)
                for i in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    x1 = candidates[i]
                    if is_safe(x1, S_minus):
                        S_plus1 = S_minus | {x1}
                        for j in range(i + 1, len(candidates)):
                            x2 = candidates[j]
                            if is_safe(x2, S_plus1):
                                # Check: |S_minus + x1 + x2| > |S_minus| = |result| - 1
                                # So we need 2 > 1, which is always true
                                result = S_plus1 | {x2}
                                improved = True
                                break
                        if improved:
                            break
                    if time.time() > deadline:
                        break
                if improved:
                    break
            if not improved:
                # Try 2-1 exchange: remove two, add one (unlikely to help but try)
                # Skip for now; 1-2 is more valuable
                pass
        
        if not improved:
            break
    
    return list(result)
