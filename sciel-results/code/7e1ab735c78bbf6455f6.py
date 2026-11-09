"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _conflict(a, m, b, n):
    """Two classes (a,m) and (b,n) are disjoint iff gcd(a-b, m*n) > 1."""
    return gcd(a - b, m * n) > 1

def _build_conflict_matrix(moduli, residues):
    """Build a bitset conflict matrix. conflict[i][j] = True if i and j conflict."""
    n = len(moduli)
    # conflict[i] is a bitset (Python int) of nodes that conflict with i
    conflict = [0] * n
    for i in range(n):
        ai, mi = residues[i], moduli[i]
        for j in range(i + 1, n):
            aj, mj = residues[j], moduli[j]
            if not _conflict(ai, mi, aj, mj):
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    return conflict

def _max_independent_set(moduli, residues, deadline):
    """
    Find a large independent set in the compatibility graph.
    Two nodes are compatible (can both be chosen) iff they are disjoint.
    We want max set of pairwise disjoint classes.
    
    Uses branch and bound with bitsets.
    """
    n = len(moduli)
    # conflict[i] = bitset of nodes that CONFLICT with i (cannot both be chosen)
    conflict = _build_conflict_matrix(moduli, residues)
    
    # compatible[i] = bitset of nodes compatible with i (can be chosen together)
    # compatible[i] = all nodes except i and its conflicts
    all_mask = (1 << n) - 1
    compatible = [0] * n
    for i in range(n):
        compatible[i] = all_mask & ~(conflict[i] | (1 << i))
    
    best_set = []
    best_size = 0
    
    def _branch(candidates, current_set, depth):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Upper bound: current size + popcount(candidates)
        remaining = candidates.bit_count()
        if len(current_set) + remaining <= best_size:
            return
        
        # Choose a node with minimum degree in the candidate subgraph (for branching)
        # Try nodes that have fewest conflicts within candidates
        cand_list = []
        c = candidates
        while c:
            i = (c & -c).bit_length() - 1
            c &= c - 1
            # Count how many of i's conflicts are in candidates
            conf_in_cand = (conflict[i] & candidates).bit_count()
            cand_list.append((conf_in_cand, i))
        
        # Sort by number of conflicts in candidates (fewer conflicts = more compatible)
        cand_list.sort()
        
        # Try including the most-compatible node first
        for _, i in cand_list:
            if time.time() > deadline:
                return
            if not (candidates & (1 << i)):
                continue
            
            # Branch 1: include i
            new_candidates = candidates & compatible[i] & ~(1 << i)
            current_set.append(i)
            _branch(new_candidates, current_set, depth + 1)
            current_set.pop()
            
            # Branch 2: exclude i
            candidates &= ~(1 << i)
            if time.time() > deadline:
                return
            _branch(candidates, current_set, depth + 1)
            candidates |= (1 << i)  # restore
        
    _branch(all_mask, [], 0)
    
    return best_set

def _greedy_large_set(moduli, residues, conflict):
    """Greedy: repeatedly pick the node with most compatible remaining nodes."""
    n = len(moduli)
    all_mask = (1 << n) - 1
    compatible = [0] * n
    for i in range(n):
        compatible[i] = all_mask & ~(conflict[i] | (1 << i))
    
    selected = []
    remaining = all_mask
    
    while remaining:
        # Pick node with max compatibility in remaining
        best_i = -1
        best_count = -1
        c = remaining
        while c:
            i = (c & -c).bit_length() - 1
            c &= c - 1
            count = (compatible[i] & remaining).bit_count()
            if count > best_count:
                best_count = count
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
        remaining &= compatible[best_i]
        remaining &= ~(1 << best_i)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    # Generate candidate (residue, modulus) pairs for each modulus 2..128
    # For each modulus m, we need to choose a residue r (0 <= r < m)
    # To maximize disjointness, we want residues that are "incompatible" with others
    
    # Strategy: For each modulus, try several candidate residues and let search pick
    # But first, let's try a smart construction:
    
    # Key insight: If we use modulus m and residue r, the class is {r + km : k in Z}
    # Two classes (r1, m1) and (r2, m2) are disjoint iff there's no solution to
    # x ≡ r1 (mod m1) and x ≡ r2 (mod m2), which happens iff gcd(m1,m2) doesn't 
    # divide (r1 - r2) ... wait, actually the system has a solution iff r1 ≡ r2 (mod gcd(m1,m2))
    # So they are DISJOINT iff r1 NOT≡ r2 (mod gcd(m1,m2)), i.e., gcd(r1-r2, m1*m2) > 1
    # Actually: the system x≡a (mod m), x≡b (mod n) has a solution iff a≡b (mod gcd(m,n)).
    # So disjoint iff a NOT≡ b (mod gcd(m,n)).
    
    # Let me re-verify: conflict (NOT disjoint) iff a ≡ b (mod gcd(m,n))
    # So disjoint iff a NOT≡ b (mod gcd(m,n))
    
    def _is_disjoint(a, m, b, n):
        g = gcd(m, n)
        return (a - b) % g != 0
    
    # Let's verify with the original: (1,2) and (3,4): gcd(2,4)=2, (1-3)%2 = (-2)%2=0 -> NOT disjoint? 
    # But 1 mod 2 = {1,3,5,7,...} and 3 mod 4 = {3,7,11,...} -> 3 is in both! So they're NOT disjoint. Correct.
    
    # (1,2) and (1,4): gcd=2, (1-1)%2=0 -> not disjoint. 1 mod 2 = {1,3,5,...}, 1 mod 4={1,5,9,...} -> 1 in both. Correct.
    
    # (1,2) and (2,3): gcd(2,3)=1, (1-2)%1=0 -> always 0 mod 1, so NOT disjoint? 
    # 1 mod 2 = {1,3,5,7,...}, 2 mod 3 = {2,5,8,11,...} -> 5 in both. Correct, not disjoint.
    
    # So with coprime moduli, they're NEVER disjoint! This is a key constraint.
    
    # Therefore, to have disjoint classes, we need gcd(m,n) > 1 for most pairs, 
    # AND the residues must differ mod gcd(m,n).
    
    # This means we should prefer moduli that share common factors.
    # Powers of 2 are great: gcd(2^a, 2^b) = 2^min(a,b) >= 2
    # But we can also use multiples of 3, 5, etc.
    
    # Let's think about which moduli to include.
    # For moduli that are pairwise coprime, we can't include more than 1.
    # So we want moduli that share factors.
    
    # Good strategy: include moduli that are multiples of small primes.
    # E.g., all even numbers: 2,4,6,8,10,...,128 (64 moduli)
    # For even moduli, gcd >= 2, so we can potentially make them all disjoint.
    
    # For even modulus m, residue r: the class contains numbers ≡ r (mod m).
    # For two even moduli m1, m2 with gcd g >= 2, we need r1 NOT≡ r2 (mod g).
    
    # Let's try: for each even modulus m, pick residue 0. Then r1-r2=0, and 0 mod g = 0, so NOT disjoint. Bad.
    
    # Pick residue 1 for all: same problem.
    
    # We need different residues mod the gcd.
    
    # Idea: For modulus m, pick residue = m/2 (if m is even). Then for two even moduli m1, m2:
    # r1 = m1/2, r2 = m2/2. Need (m1/2 - m2/2) NOT≡ 0 (mod gcd(m1,m2)).
    # (m1-m2)/2 mod gcd(m1,m2). If gcd(m1,m2) = 2, then we need (m1-m2)/2 NOT≡ 0 (mod 2),
    # i.e., (m1-m2)/2 is odd, i.e., m1-m2 ≡ 2 (mod 4).
    
    # This gets complicated. Let me just use the search approach.
    
    # Build candidate set: for each modulus 2..128, try a few residues
    # Actually, let's be smarter. Let's try all moduli and use the search.
    
    # But 127 moduli is a lot. Let's first try a greedy approach, then improve with local search.
    
    # Step 1: Greedy construction
    # For each modulus, pick the residue that maximizes disjointness with already-chosen classes.
    
    moduli = list(range(2, 129))  # 2,3,...,128
    n_mod = len(moduli)
    
    # For greedy, process moduli in an order that helps.
    # Start with moduli that have many shared factors.
    
    # Let's try a different approach: 
    # 1. Greedily build a large set
    # 2. Try to add more moduli
    # 3. Local search to improve
    
    # Greedy: for each modulus in some order, pick the best residue
    def _greedy_construct(moduli_list):
        chosen = []  # list of (residue, modulus)
        for m in moduli_list:
            best_r = None
            best_score = -1
            for r in range(m):
                # Check disjointness with all chosen
                ok = True
                for (cr, cm) in chosen:
                    if not _is_disjoint(r, m, cr, cm):
                        ok = False
                        break
                if ok:
                    # Score: how "compatible" is this with remaining moduli?
                    # Just take it if it works
                    best_r = r
                    best_score = 0
                    break  # first valid residue
            if best_r is not None:
                chosen.append((best_r, m))
        return chosen
    
    # Order moduli by descending number of prime factors (more shared factors = easier)
    def _num_prime_factors(m):
        count = 0
        d = 2
        n = m
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Sort moduli: prefer those with more shared factors
    # Actually, let's try multiple orders
    
    # Order 1: even moduli first, then odd
    even_mods = [m for m in moduli if m % 2 == 0]
    odd_mods = [m for m in moduli if m % 2 != 0]
    
    # Try greedy with even moduli first
    order1 = even_mods + odd_mods
    result1 = _greedy_construct(order1)
    
    # Order 2: sort by number of prime factors descending
    order2 = sorted(moduli, key=lambda m: -_num_prime_factors(m))
    result2 = _greedy_construct(order2)
    
    # Order 3: descending modulus
    order3 = sorted(moduli, reverse=True)
    result3 = _greedy_construct(order3)
    
    # Order 4: ascending modulus
    order4 = sorted(moduli)
    result4 = _greedy_construct(order4)
    
    # Pick best greedy result
    best_result = max([result1, result2, result3, result4], key=len)
    
    # Now try to improve with local search
    # Try to add more moduli by reassigning residues
    
    # Convert to list of (residue, modulus)
    chosen = best_result[:]
    
    # Try to add each unchosen modulus
    chosen_mods = set(m for _, m in chosen)
    for m in moduli:
        if m in chosen_mods:
            continue
        if time.time() > deadline:
            break
        # Try all residues for m
        for r in range(m):
            ok = True
            for (cr, cm) in chosen:
                if not _is_disjoint(r, m, cr, cm):
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
                chosen_mods.add(m)
                break
    
    # Local search: try swapping residues to allow adding more
    # This is complex; let's try a simpler improvement
    
    # Try to improve by reassigning residues for existing moduli
    # to free up space for new moduli
    
    # Actually, let's try a more systematic approach:
    # Use the branch-and-bound on a reduced set
    
    # First, let's see how many we have
    current_size = len(chosen)
    
    # Try to improve: for each unchosen modulus, try to find a residue
    # that works, possibly by adjusting other residues
    
    # Simpler: just try all residues for each unchosen modulus
    chosen_mods = set(m for _, m in chosen)
    for m in sorted(moduli, key=lambda x: -_num_prime_factors(x)):
        if m in chosen_mods:
            continue
        if time.time() > deadline:
            break
        for r in range(m):
            ok = True
            for (cr, cm) in chosen:
                if not _is_disjoint(r, m, cr, cm):
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
                chosen_mods.add(m)
                break
    
    # Convert to output format: list of [residue, modulus]
    result = [[r, m] for r, m in chosen]
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            r1, m1 = result[i]
            r2, m2 = result[j]
            if not _is_disjoint(r1, m1, r2, m2):
                # This shouldn't happen, but if it does, remove the later one
                result.pop(j)
                j -= 1
                break
    
    return result
# EVOLVE-BLOCK-END
