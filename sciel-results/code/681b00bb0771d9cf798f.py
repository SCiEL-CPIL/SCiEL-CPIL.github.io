"""Maximize pairwise-disjoint residue classes using CSP with MRV and Forward Checking."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

MAX_MOD = 128
DEADLINE = 85.0  # Safety margin before 90s kill

def _is_pairwise_disjoint(classes):
    """Exact gcd criterion: classes (r,m),(r',m') disjoint iff gcd(m,m') ∤ (r-r')."""
    n = len(classes)
    for i in range(n):
        r, m = classes[i]
        for j in range(i + 1, n):
            r2, m2 = classes[j]
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
    return True

def _is_valid(classes):
    if not classes:
        return False
    seen = set()
    for r, m in classes:
        if not (2 <= m <= MAX_MOD):
            return False
        if not (0 <= r < m):
            return False
        if m in seen:
            return False
        seen.add(m)
    return _is_pairwise_disjoint(classes)

def _greedy_seed():
    """
    Generate a strong initial seed using a deterministic greedy approach.
    We prioritize moduli with many small prime factors or just iterate in a smart order.
    Actually, a simple greedy over sorted moduli often works well if we pick the 'easiest' residue first.
    """
    # Precompute gcds
    gcd_cache = {}
    for m1 in range(2, MAX_MOD + 1):
        for m2 in range(m1, MAX_MOD + 1):
            gcd_cache[(m1, m2)] = gcd(m1, m2)
            gcd_cache[(m2, m1)] = gcd(m1, m2)
            
    # Order moduli by number of divisors or just natural order?
    # Let's try natural order first, but maybe prime moduli are easier to place?
    # No, prime moduli have gcd 1 with everything, so they are ALWAYS compatible with any residue.
    # Wait, if gcd(m, m') = 1, then (r - r') % 1 == 0 is always true? 
    # NO. The condition for intersection is (r - r') % gcd(m, m') == 0.
    # If gcd = 1, then (r - r') % 1 == 0 is always TRUE. 
    # So if gcd(m, m') = 1, the classes ALWAYS intersect?
    # Let's re-read the criterion: "disjoint iff gcd(m,m') ∤ (r-r')"
    # If gcd=1, 1 divides everything. So gcd(m,m') | (r-r') is ALWAYS true.
    # Therefore, if gcd(m, m') = 1, the classes are NEVER disjoint.
    # This is a crucial insight!
    # Two residue classes (r, m) and (r', m') are disjoint iff gcd(m, m') does NOT divide (r - r').
    # If gcd(m, m') = 1, then 1 divides (r - r'), so they are NOT disjoint.
    # Thus, we CANNOT have two moduli that are coprime in our set.
    # All moduli in our set must share a common factor > 1 with each other?
    # No, just pairwise. For any pair (m_i, m_j), gcd(m_i, m_j) must be > 1 AND we must choose residues such that r_i != r_j mod gcd(m_i, m_j).
    
    # This changes the problem significantly. We are looking for a clique in a graph where edges exist between moduli if gcd(m_i, m_j) > 1, 
    # and then we must assign residues such that for every edge, the residues are different modulo the gcd.
    
    # Let's verify: 
    # Classes A: x ≡ r1 (m1), B: x ≡ r2 (m2).
    # Intersection exists iff r1 ≡ r2 (gcd(m1, m2)).
    # Disjoint iff r1 != r2 (mod gcd(m1, m2)).
    # If gcd(m1, m2) = 1, then r1 ≡ r2 (1) is always true. So they always intersect.
    # So yes, all moduli must be pairwise non-coprime.
    
    # This means we should restrict our candidate moduli to those that form a "dense" non-coprime graph.
    # E.g., all even numbers. gcd(even, even) >= 2.
    # If we pick all even moduli, gcd(m, m') is at least 2.
    # We need to assign residues such that r_i != r_j mod gcd(m_i, m_j).
    
    # Let's restart the seed with this insight.
    # A good set of moduli is all multiples of a small prime, e.g., 2.
    # Or a mix of multiples of 2 and 3, but they must be non-coprime pairwise.
    # Multiples of 2: 2, 4, ..., 128.
    # Multiples of 3: 3, 6, 9, ...
    # If we include 2 and 3, gcd(2,3)=1 -> Bad.
    # So we can't mix 2 and 3 unless we include 6? No, pairwise condition.
    # If we have 2 and 3 in the set, gcd(2,3)=1 -> Fail.
    # So the set of moduli must be such that every pair has gcd > 1.
    # This is a "pairwise intersecting" family of sets of prime factors.
    # By Erdos-Ko-Rado type theorems, large such families are often stars (all share a common prime) or triangles (for small sets).
    # For moduli up to 128, the largest family likely shares a common prime factor, e.g., 2.
    # Let's check if we can do better with a non-star.
    # E.g., {6, 10, 15}. gcd(6,10)=2, gcd(6,15)=3, gcd(10,15)=5. All > 1.
    # Can we add 2? gcd(2,6)=2, gcd(2,10)=2, gcd(2,15)=1. Fail.
    # Can we add 3? gcd(3,6)=3, gcd(3,10)=1. Fail.
    # Can we add 5? gcd(5,6)=1. Fail.
    # So {6, 10, 15} is a clique of size 3.
    # Star of 2: {2, 4, 6, ..., 128}. Size 63.
    # Star of 3: {3, 6, 9, ..., 126}. Size 42.
    # So the star of 2 is much larger.
    # It is highly likely the optimal solution uses all even moduli or a large subset of them.
    
    # Let's implement a CSP solver that focuses on the even moduli first, but allows others if they fit.
    # Actually, if we include an odd number, it must be non-coprime with ALL other moduli.
    # If we have the star of 2, any odd number m must be non-coprime with 2. Impossible.
    # So if we have 2 in the set, we cannot have any odd numbers.
    # If we drop 2, can we have more?
    # Suppose we use multiples of 4? {4, 8, ..., 128}. Size 32.
    # We can add 6? gcd(4,6)=2, gcd(8,6)=2... gcd(128,6)=2. Yes.
    # So {4, 8, ..., 128} U {6, 10, 14, ...}?
    # Let's check 4 and 6. gcd=2.
    # 4 and 10. gcd=2.
    # 8 and 6. gcd=2.
    # It seems we can combine multiples of 4 with even numbers that are not multiples of 4?
    # Let S be the set of even moduli.
    # If m, m' are even, gcd(m, m') >= 2.
    # So the set of ALL even moduli is a valid clique in terms of gcd > 1.
    # Size = 63.
    # Can we add any odd modulus?
    # If m is odd, it must be non-coprime with 2. gcd(odd, 2) = 1.
    # So if 2 is in the set, no odd numbers.
    # What if we exclude 2?
    # Set: {4, 6, 8, 10, ..., 128}.
    # Does this allow odd numbers?
    # If we add 3. gcd(3, 4)=1. Fail.
    # If we add 9. gcd(9, 4)=1. Fail.
    # It seems any odd number is coprime to 4, 8, 16, etc.
    # So if we have any power of 2 > 2, we can't have odd numbers.
    # What if we use only {6, 10, 14, ...} (2 * odd)?
    # gcd(6, 10) = 2.
    # Can we add 3? gcd(3, 6)=3, gcd(3, 10)=1. Fail.
    # Can we add 9? gcd(9, 6)=3, gcd(9, 10)=1. Fail.
    
    # Conclusion: The largest clique of moduli is almost certainly the set of all even numbers.
    # Size 63.
    # Now, can we assign residues to all 63 even moduli such that they are pairwise disjoint?
    # Condition: For any m, m' in Even, r != r' mod gcd(m, m').
    # Since gcd(m, m') is always even, let g = gcd(m, m').
    # We need r != r' mod g.
    # This is a system of constraints.
    # Let's try to assign residues greedily or via CSP.
    
    # Strategy:
    # 1. Fix the set of moduli to be the even numbers 2..128.
    # 2. Use CSP with MRV to assign residues.
    # 3. If CSP fails, try removing moduli that cause conflicts (local search on the set).
    # 4. Also try other large cliques (e.g. multiples of 3, or mixed) just in case, but even is likely best.
    
    even_mods = [m for m in range(2, MAX_MOD + 1) if m % 2 == 0]
    
    # Precompute gcds for even mods
    # We will use a CSP solver.
    
    return even_mods

def _solve_csp(moduli, deadline):
    """
    Solve the CSP: assign residue r_m to each m in moduli.
    Constraints: For all i != j, r_i != r_j mod gcd(m_i, m_j).
    """
    n = len(moduli)
    if n == 0:
        return {}
    
    # Precompute gcds and constraint lists
    # constraints[i] is a list of (j, g) meaning r_i != r_j mod g
    constraints = [[] for _ in range(n)]
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            constraints[i].append((j, g))
            constraints[j].append((i, g))
            
    # Domains: initially all residues 0..m-1
    domains = [set(range(m)) for m in moduli]
    
    # MRV Heuristic: select variable with smallest domain
    # We need to maintain domains dynamically.
    # For speed, we'll use a simple backtracking with forward checking.
    
    assignment = {}
    
    # Sort moduli? MRV will handle order dynamically.
    # But initial order can help.
    
    def find_mrv():
        min_size = float('inf')
        min_idx = -1
        for i in range(n):
            if i in assignment:
                continue
            size = len(domains[i])
            if size < min_size:
                min_size = size
                min_idx = i
            if size == 0:
                return -1 # Failure
        return min_idx
        
    def forward_check(idx, val):
        # Propagate val to neighbors
        new_domains = [d.copy() for d in domains]
        for j, g in constraints[idx]:
            if j in assignment:
                continue
            # Remove values v from new_domains[j] such that v == val mod g
            # v % g == val % g
            target = val % g
            # Filter
            # Since g can be large, iterating over all v in domain is O(m).
            # We can optimize by checking only if domain has values in that residue class.
            # For now, simple filter.
            new_domains[j] = {v for v in new_domains[j] if v % g != target}
            if not new_domains[j]:
                return None # Failure
        return new_domains
        
    def backtrack(assignment, domains):
        if len(assignment) == n:
            return True
            
        if time.time() > deadline:
            return False # Timeout, treat as failure for this branch, but we want best partial?
            # Actually, if we timeout, we should return whatever we have so far?
            # The function returns bool. We need to handle timeout carefully.
            # Let's raise an exception or return a special value.
            # For now, let's just stop searching.
            
        idx = find_mrv()
        if idx == -1:
            return False
            
        # Try values in domain
        # Heuristic: try values that are "central" or random?
        # Random might help escape symmetry.
        vals = list(domains[idx])
        random.shuffle(vals)
        
        for val in vals:
            new_domains = forward_check(idx, val)
            if new_domains is None:
                continue
            assignment[idx] = val
            if backtrack(assignment, new_domains):
                return True
            del assignment[idx]
            
        return False

    # Start
    success = backtrack(assignment, domains)
    if success:
        return {moduli[i]: assignment[i] for i in range(n)}
    else:
        # Return partial assignment if possible?
        # The task requires a valid list. If we can't solve fully, we should return a valid subset.
        # We can extract the largest consistent subset from the partial assignment?
        # Or just return empty and let the caller handle it.
        # Better: return the assignment we have so far if it's consistent?
        # The assignment dict only contains consistent values for the assigned vars.
        # But we need to ensure the unassigned vars don't break it? No, they are just not included.
        # So we can return the partial assignment as a valid set of classes.
        return {moduli[i]: assignment[i] for i in assignment}

def construct():
    t0 = time.time()
    
    # 1. Identify candidate moduli sets
    # Primary candidate: All even moduli
    even_mods = [m for m in range(2, MAX_MOD + 1) if m % 2 == 0]
    
    # Secondary candidates: Multiples of 3, etc. (less likely to be better, but good for diversity)
    # Also, maybe we can find a larger set by dropping some evens and adding odds?
    # Unlikely given the gcd(2, odd)=1 constraint if 2 is present.
    # If we drop 2, can we add 3?
    # Set: Even \ {2} U {3}?
    # 3 and 4 (gcd 1) -> Fail.
    # So 3 cannot coexist with 4.
    # It seems the even set is very robust.
    
    best_solution = []
    
    # Try solving CSP on even moduli
    # To speed up, we might need to prune the moduli set if CSP is too slow.
    # But 63 variables is small.
    
    # Optimization: We can pre-check if a simpler assignment works.
    # e.g., r_m = 0 for all m?
    # If r_m = 0, then r_i - r_j = 0.
    # Condition: gcd(m_i, m_j) ∤ 0.
    # gcd always divides 0. So r=0 for all is NEVER valid (unless no pairs).
    # We need r_i != r_j mod gcd.
    
    # Let's run the CSP solver.
    # We give it a deadline of 40s.
    deadline_csp = t0 + 40.0
    
    solution = _solve_csp(even_mods, deadline_csp)
    
    if solution:
        classes = [[r, m] for m, r in solution.items()]
        if _is_valid(classes):
            best_solution = classes
    
    # If the even set didn't yield a full solution or timed out, try other sets or local search on the result.
    # Also, try a local search to improve the count if we got a partial solution.
    
    # Local Search on the current best
    if best_solution:
        # Convert to list of [r, m]
        current = best_solution
        current_mods = set(m for r, m in current)
        
        # Precompute gcds for all moduli 2..128
        gcd_cache = [[0]*(MAX_MOD+1) for _ in range(MAX_MOD+1)]
        for m1 in range(2, MAX_MOD+1):
            for m2 in range(m1, MAX_MOD+1):
                g = gcd(m1, m2)
                gcd_cache[m1][m2] = g
                gcd_cache[m2][m1] = g
                
        def is_valid_state(clique):
            n = len(clique)
            for i in range(n):
                r1, m1 = clique[i]
                for j in range(i+1, n):
                    r2, m2 = clique[j]
                    g = gcd_cache[m1][m2]
                    if (r1 - r2) % g == 0:
                        return False
            return True

        # Try to add more moduli or swap
        # Since we likely have all evens, there are no unused evens.
        # Can we add odds?
        # Only if we remove some evens that are coprime to the odds.
        # This is a complex swap.
        
        # Let's just do a simple local search to see if we can improve the partial solution.
        # If we have a full solution of 63, we can't improve.
        # If we have < 63, we can try to add back moduli we dropped or swap.
        
        if len(best_solution) < len(even_mods):
             # Try to add missing evens
             missing = [m for m in even_mods if m not in current_mods]
             random.shuffle(missing)
             for m_new in missing:
                 if time.time() > t0 + 80:
                     break
                 # Try to find a residue for m_new
                 valid_res = []
                 for r in range(m_new):
                     ok = True
                     for r2, m2 in current:
                         g = gcd_cache[m_new][m2]
                         if (r - r2) % g == 0:
                             ok = False
                             break
                     if ok:
                         valid_res.append(r)
                 if valid_res:
                     r_new = random.choice(valid_res)
                     new_clique = current + [[r_new, m_new]]
                     if is_valid_state(new_clique):
                         current = new_clique
                         current_mods.add(m_new)
        
        best_solution = current

    # Fallback: If CSP failed completely or returned empty, use a greedy seed
    if not best_solution:
        # Simple greedy on evens
        classes = []
        for m in even_mods:
            for r in range(m):
                ok = True
                for r2, m2 in classes:
                    g = gcd(m, m2)
                    if (r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    classes.append([r, m])
                    break
        best_solution = classes
        
    # Final validation
    if _is_valid(best_solution):
        return best_solution
    else:
        # Return a minimal valid set
        return [[0, 2]]

# EVOLVE-BLOCK-END
