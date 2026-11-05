import time
import math

def construct():
    # Deadline guard: must finish well before 90s wall-clock.
    t_start = time.time()
    DEADLINE = 85.0  # seconds, safety margin
    
    n = 256
    
    # The current best known approach uses the "large interval" construction:
    # S = {k, k+1, ..., n} where k is chosen so that for all a in S,
    # the range of possible b+c (with a < b <= c in S) contains no multiple of a.
    # 
    # For a in S, b+c ranges from (a+1)+(a+2) = 2a+3 to n+(n) = 2n = 512.
    # Actually more carefully: b > a, c >= b, so min(b+c) = (a+1)+(a+2) = 2a+3
    # and max(b+c) = n + n = 2n.
    # We need: no multiple of a in [2a+3, 2n].
    # Multiples of a in that range: a*m for m >= 3 (since a*2 = 2a < 2a+3)
    # up to a*m <= 2n => m <= 2n/a.
    # So we need: the smallest m >= 3 such that a*m > 2n, i.e., a*3 > 2n.
    # That gives a > 2n/3 = 512/3 ≈ 170.67, so a >= 171.
    #
    # With k=171: S = {171, ..., 256}, size = 256-171+1 = 86.
    # Score = 86 / seed_size. Seed score is 1.0, and the current program scores 1.3231,
    # implying seed size = 86/1.3231 ≈ 65.0... Let me check: 1.3231 * 65 = 86.0. Yes.
    # So seed has 65 elements.
    
    # Can we do better than 86? Let's think about whether we can include some smaller elements.
    # 
    # The constraint is: for all a < b <= c in S, a does NOT divide b+c.
    # 
    # Key insight: if we include a small element a, it constrains many pairs (b,c).
    # But maybe we can include some elements below 171 if we remove conflicting ones.
    #
    # Let's try a greedy/local search approach:
    # Start with the full interval [171, 256] (size 86), then try to add elements below 171.
    # For each candidate x < 171, check if adding x to S violates any constraint.
    # A constraint is violated if there exist b, c in S with x < b <= c and x | (b+c).
    #
    # Actually, let's think more carefully. If we add x to S:
    # - For pairs (b,c) both in original S: need x not to divide b+c for all b,c in S with x < b <= c.
    #   Since x < 171 <= b, this is always x < b, so we need: for all b,c in S with b<=c, x ∤ (b+c).
    # - For pairs where one of them is x: but x is the smallest, so x < b always. We need x ∤ (b+c) where b,c in S.
    #   Wait, the constraint is for a < b <= c. If a=x, then we need x ∤ (b+c) for all b,c in S with x < b <= c.
    #   Since all elements in S are >= 171 > x, this is: for all b,c in S with b <= c, x ∤ (b+c).
    #
    # Also, if we add multiple small elements, they interact with each other.
    #
    # Let's try a different approach: use a greedy algorithm that builds S from large to small,
    # checking feasibility at each step.
    
    # First, let's verify the base case
    base_S = list(range(171, 257))
    
    # Try to add elements from 170 down to 1, one at a time
    # For efficiency, precompute which pairs in S have b+c divisible by candidate x
    
    def is_valid_with(S, x):
        """Check if adding x to S maintains the property."""
        # x must be less than all elements in S (since we're adding smaller elements)
        # For all b, c in S with x < b <= c: x must not divide b+c
        # Since x < 171 <= all elements in S, x < b is automatic.
        # So we need: for all b, c in S with b <= c, x ∤ (b+c)
        for i, b in enumerate(S):
            for c in S[i:]:
                if (b + c) % x == 0:
                    return False
        return True
    
    # Greedy: try to add each element from 170 down to 1
    S = list(base_S)
    
    # Precompute sums for efficiency
    # Actually, let's be smart: for a given x, we need to check if any pair (b,c) in S
    # has b+c ≡ 0 (mod x). This is equivalent to: for each residue r mod x,
    # if there are two elements in S with residues r and (x-r) mod x (or one element with residue 0
    # appearing twice, i.e., b=c with b≡0 mod x).
    
    def can_add(S, x):
        """Check if x can be added to S without violating constraints.
        For all b, c in S with x < b <= c: x must not divide b+c."""
        # Since we're adding x which is smaller than all elements in S,
        # we need: for all b, c in S with b <= c, x does not divide (b+c)
        # But we also need x < b, which is satisfied since x < min(S)
        
        # Efficient check using residues
        # Group elements by residue mod x
        from collections import defaultdict
        residue_groups = defaultdict(list)
        for v in S:
            r = v % x
            residue_groups[r].append(v)
        
        # For each pair of residues (r1, r2) where (r1 + r2) % x == 0,
        # check if there exist b, c with b <= c and b % x = r1, c % x = r2
        # such that b+c is divisible by x. This is automatically true if residues sum to 0 mod x.
        # So we just need to check if such a pair exists.
        
        for r1 in residue_groups:
            r2 = (x - r1) % x
            if r1 == r2:
                # Need at least 2 elements with this residue (for b=c or b<c)
                if len(residue_groups[r1]) >= 2:
                    return False
            else:
                if r2 in residue_groups:
                    # Need at least one element from each group
                    # And we need b <= c, which is always possible if both groups non-empty
                    if len(residue_groups[r1]) >= 1 and len(residue_groups[r2]) >= 1:
                        # Verify: pick smallest from one, largest from other, check b <= c
                        # Actually, since all elements in S are > x, and we just need existence
                        # of some b <= c with the right residues, this should work.
                        # But let's be safe: check if min of larger group <= max of smaller group
                        # Actually, we can always pair them: take any b from group r1 and any c from group r2
                        # If b > c, swap. So we need at least one pair where after sorting, b <= c.
                        # This is always true if both groups have elements.
                        return False
        return True
    
    # Try adding elements from 170 down to 1
    for x in range(170, 0, -1):
        if time.time() - t_start > DEADLINE:
            break
        if can_add(S, x):
            S.append(x)
            S.sort()
    
    # Now try local search: remove some elements and add others
    # But first, let's see what we have
    
    # Actually, the greedy might not be optimal. Let's try a more systematic approach.
    # 
    # Alternative: formulate as a maximum independent set problem on a conflict graph.
    # Two elements a, b conflict if... no, the constraint is on triples.
    # 
    # The constraint is: for all a < b <= c in S, a ∤ (b+c).
    # This is a 3-uniform hypergraph constraint. Maximum independent set in 3-uniform
    # hypergraphs is NP-hard, but for n=256 we might be able to do well with heuristics.
    #
    # Let's try: start with the base solution, then do local search by swapping elements.
    
    # Convert to a set for fast lookup
    S_set = set(S)
    
    def count_violations(S_list):
        """Count the number of violating triples in S_list."""
        S_list = sorted(S_list)
        violations = 0
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for c in S_list[j:]:
                    if (b + c) % a == 0:
                        violations += 1
        return violations
    
    # Verify our current solution is valid
    if count_violations(S) > 0:
        # Fall back to base
        S = list(base_S)
        S_set = set(S)
    
    # Local search: try to improve by removing one element and adding another
    best_S = list(S)
    best_size = len(S)
    
    # Try single swaps: remove x from S, add y not in S
    # For each x in S, for each y not in S with y < 171 (or any y), check if swap improves
    # This is expensive, so let's be selective.
    
    # Actually, let's try a different strategy: 
    # Start from scratch with a greedy that considers elements in a smart order.
    
    # Key observation: elements > 2n/3 are "safe" (they don't conflict with each other).
    # Elements in (n/2, 2n/3] might conflict with some but not all large elements.
    # Elements <= n/2 have more conflicts.
    
    # Let's try: include all elements from k to n, then greedily add smaller elements.
    # But also try different values of k.
    
    best_solution = list(base_S)
    best_score = len(base_S)
    
    # Try different starting points k
    for k in range(160, 180):
        if time.time() - t_start > DEADLINE:
            break
        candidate = list(range(k, n + 1))
        # Now try to add elements below k
        for x in range(k - 1, 0, -1):
            if time.time() - t_start > DEADLINE:
                break
            if can_add(candidate, x):
                candidate.append(x)
                candidate.sort()
        
        if len(candidate) > best_score:
            best_score = len(candidate)
            best_solution = list(candidate)
    
    # Also try the original greedy from 171
    if len(S) > best_score:
        best_score = len(S)
        best_solution = list(S)
    
    # Try local search on best_solution
    S_ls = list(best_solution)
    S_ls_set = set(S_ls)
    
    # Try to add more elements
    for x in range(1, n + 1):
        if time.time() - t_start > DEADLINE:
            break
        if x not in S_ls_set:
            if can_add(S_ls, x):
                S_ls.append(x)
                S_ls_set.add(x)
                S_ls.sort()
    
    if len(S_ls) > best_score:
        best_score = len(S_ls)
        best_solution = list(S_ls)
    
    # Local search with swaps: try removing one element and adding another
    improved = True
    while improved and time.time() - t_start < DEADLINE:
        improved = False
        current = list(best_solution)
        current_set = set(current)
        
        # Try all possible swaps: remove x from current, add y not in current
        for i, x in enumerate(current):
            if time.time() - t_start > DEADLINE:
                break
            # Temporarily remove x
            trial = current[:i] + current[i+1:]
            trial_set = set(trial)
            
            # Try adding each candidate y not in trial
            for y in range(1, n + 1):
                if y in trial_set:
                    continue
                if can_add(trial, y):
                    # This swap is valid and maintains size
                    # But we want to see if it enables further improvements
                    new_solution = trial + [y]
                    new_solution.sort()
                    if len(new_solution) > len(current):
                        # Shouldn't happen with swap, but check
                        best_solution = new_solution
                        best_score = len(new_solution)
                        improved = True
                        break
                    # Even if same size, try it as it might enable more additions
                    # Try greedy add on this new solution
                    temp = list(new_solution)
                    temp_set = set(temp)
                    for z in range(1, n + 1):
                        if time.time() - t_start > DEADLINE:
                            break
                        if z not in temp_set:
                            if can_add(temp, z):
                                temp.append(z)
                                temp_set.add(z)
                                temp.sort()
                    if len(temp) > len(current):
                        best_solution = temp
                        best_score = len(temp)
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue
    
    # Final verification
    S_final = sorted(best_solution)
    if count_violations(S_final) > 0:
        # Fallback
        S_final = list(base_S)
    
    # One more pass: try to add any remaining elements
    S_final_set = set(S_final)
    for x in range(1, n + 1):
        if time.time() - t_start > DEADLINE:
            break
        if x not in S_final_set:
            if can_add(S_final, x):
                S_final.append(x)
                S_final_set.add(x)
                S_final.sort()
    
    # Final final verification
    S_final = sorted(S_final)
    if count_violations(S_final) > 0:
        # Fallback to base
        S_final = list(base_S)
    
    return S_final
