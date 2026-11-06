"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

_START_TIME = None


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    deadline = _START_TIME + 85.0  # safety margin before 90s kill

    moduli = list(range(2, 129))
    n = len(moduli)

    # Precompute gcd table
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_table[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Key insight: if gcd(m1, m2) == 1, the classes ALWAYS intersect.
    # So we can only pick moduli that are pairwise non-coprime.
    # Find the largest clique in the "non-coprime" graph.
    # But even within a clique, we need to check residue compatibility.
    
    # First, find all pairs that are non-coprime
    non_coprime = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd_table[i][j] > 1:
                non_coprime[i][j] = True
                non_coprime[j][i] = True

    # Find maximal cliques in the non-coprime graph using a simple approach
    # For 127 nodes, we can use a greedy + backtracking approach.
    # But first, let's find connected components and work within them.
    
    # Actually, let's use a different approach:
    # The moduli 2..128. Many share factor 2, 3, 5, 7, etc.
    # Let's identify the structure:
    # - All even numbers share factor 2
    # - All multiples of 3 share factor 3
    # - etc.
    # 
    # A set of moduli is pairwise non-coprime if for every pair, gcd > 1.
    # This is equivalent to: the set forms a clique in the non-coprime graph.
    
    # Strategy: Use backtracking to find a large clique, then for that clique,
    # find residues that make all classes disjoint.
    
    # But actually, we can combine both: do a backtracking search where at each
    # step we pick a modulus AND a residue, checking disjointness with all previous.
    
    # For speed, precompute for each pair (i,j) with gcd>1, the constraint:
    # (r_j - r_i) % gcd(m_i, m_j) != 0
    
    # Let's use a smarter approach:
    # 1. Find a large pairwise-non-coprime subset of moduli (clique in non-coprime graph)
    # 2. For that subset, use constraint propagation to find compatible residues
    # 3. If stuck, backtrack and try a different clique
    
    # Step 1: Find large cliques
    # Use a simple greedy: sort by degree (number of non-coprime neighbors),
    # pick highest degree first.
    
    degree = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if non_coprime[i][j]:
                degree[i] += 1
                degree[j] += 1

    # Greedy clique finding
    def greedy_clique(order):
        clique = []
        for idx in order:
            ok = True
            for c in clique:
                if not non_coprime[idx][c]:
                    ok = False
                    break
            if ok:
                clique.append(idx)
        return clique

    # Try multiple orderings
    best_clique = []
    orderings = [
        sorted(range(n), key=lambda i: (-degree[i], -moduli[i])),
        sorted(range(n), key=lambda i: (-degree[i], moduli[i])),
        sorted(range(n), key=lambda i: (-moduli[i], -degree[i])),
        sorted(range(n), key=lambda i: (moduli[i], -degree[i])),
    ]
    for order in orderings:
        c = greedy_clique(order)
        if len(c) > len(best_clique):
            best_clique = c

    # Also try randomized greedy
    for _ in range(200):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
        # Bias toward high degree
        order.sort(key=lambda i: (-degree[i] * 0.7 - random.random() * 0.3, moduli[i]))
        c = greedy_clique(order)
        if len(c) > len(best_clique):
            best_clique = c

    # Now, for the best clique, find compatible residues using backtracking
    # The clique size might be large (e.g., all evens: 63 numbers).
    # For 63 moduli, we need to assign residues such that for every pair (i,j),
    # (r_j - r_i) % gcd(m_i, m_j) != 0.
    
    # This is a constraint satisfaction problem. Use backtracking with MRV.
    
    def solve_residues(clique, time_limit=30.0):
        """Find residues for all moduli in clique such that all classes are disjoint."""
        k = len(clique)
        if k == 0:
            return []
        
        # For each modulus in clique, the domain is 0..m-1
        # Constraints: for pair (i,j), (r_j - r_i) % gcd(m_i, m_j) != 0
        
        # Precompute constraints
        # For each pair, store (i, j, g) where g = gcd(m_i, m_j)
        constraints = []
        for a in range(k):
            for b in range(a + 1, k):
                g = gcd_table[clique[a]][clique[b]]
                if g > 1:
                    constraints.append((a, b, g))
        
        # MRV: at each step, pick the variable with fewest remaining values
        # For initial assignment, all have full domain.
        # Use forward checking: when assigning r_a, eliminate values for unassigned vars.
        
        # For efficiency with large k, use a different approach:
        # Note that if all moduli share a common factor p (e.g., all even),
        # then gcd(m_i, m_j) is at least p for all pairs.
        # The constraint (r_j - r_i) % g != 0 means r_j % g != r_i % g.
        
        # Key insight: if all moduli are multiples of p, then for any pair,
        # gcd >= p. The constraint is (r_j - r_i) % g != 0.
        # Since g >= p, this implies (r_j - r_i) % p != 0 as well? No, not necessarily.
        # (r_j - r_i) % g != 0 does NOT imply (r_j - r_i) % p != 0.
        # Example: g=6, p=2. r_j - r_i = 3. 3%6 != 0 (ok), but 3%2 = 1 != 0 (also ok).
        # Example: g=6, p=2. r_j - r_i = 2. 2%6 != 0 (ok), 2%2 = 0.
        # So the p-constraint is weaker.
        
        # Let's just do backtracking with forward checking.
        # For large k, this might be slow. Let's try a smarter approach.
        
        # Alternative: if all moduli in the clique are even, then gcd(m_i, m_j) >= 2.
        # The constraint (r_j - r_i) % gcd >= 2 means r_j % 2 != r_i % 2? No.
        # (r_j - r_i) % g != 0 where g >= 2.
        # If g = 2: r_j % 2 != r_i % 2.
        # If g = 4: r_j % 4 != r_i % 4.
        # If g = 6: r_j % 6 != r_i % 6.
        
        # The tightest constraint is the one with the smallest g.
        # If all pairs have gcd >= 2, then at minimum, for pairs with gcd=2,
        # we need r_j % 2 != r_i % 2.
        
        # Let's try: assign residues greedily, checking constraints as we go.
        # Order: process moduli in order of increasing modulus (smaller modulus = fewer choices).
        
        clique_sorted = sorted(clique, key=lambda i: moduli[i])
        k = len(clique_sorted)
        
        # domains[i] = set of valid residues for clique_sorted[i]
        # Start with full domain
        domains = [list(range(moduli[clique_sorted[i]])) for i in range(k)]
        
        # Assign in order
        assignment = [None] * k
        
        def backtrack(pos):
            if time.time() > time_limit:
                return False
            if pos == k:
                return True
            
            # Try each value in domain
            for r in domains[pos]:
                # Check against all previously assigned
                ok = True
                for prev in range(pos):
                    g = gcd_table[clique_sorted[pos]][clique_sorted[prev]]
                    if g > 1 and (r - assignment[prev]) % g == 0:
                        ok = False
                        break
                if not ok:
                    continue
                
                assignment[pos] = r
                
                # Forward checking: prune domains of future variables
                pruned = []
                feasible = True
                for fut in range(pos + 1, k):
                    g = gcd_table[clique_sorted[pos]][clique_sorted[fut]]
                    if g > 1:
                        new_domain = [v for v in domains[fut] if (v - r) % g != 0]
                        if not new_domain:
                            feasible = False
                            break
                        pruned.append((fut, new_domain))
                    else:
                        pruned.append((fut, None))  # no change
                    # Actually, if g==1, no constraint, domain unchanged
                    # But wait, g==1 means always intersect, which shouldn't happen in our clique
                
                if not feasible:
                    assignment[pos] = None
                    continue
                
                # Apply pruning
                saved_domains = []
                for fut, new_dom in pruned:
                    if new_dom is not None:
                        saved_domains.append((fut, domains[fut][:]))
                        domains[fut] = new_dom
                
                if backtrack(pos + 1):
                    return True
                
                # Undo pruning
                for fut, old_dom in saved_domains:
                    domains[fut] = old_dom
                
                assignment[pos] = None
            
            return False
        
        if backtrack(0):
            return [[assignment[i], moduli[clique_sorted[i]]] for i in range(k)]
        
        return None

    # Try solving for the best clique
    best_solution = None
    best_size = 0
    
    # Try the greedy clique
    if best_clique:
        sol = solve_residues(best_clique, time_limit=20.0)
        if sol:
            best_solution = sol
            best_size = len(sol)

    # Try smaller subsets if the full clique fails
    # Also try different cliques
    all_cliques = [best_clique]
    
    # Generate more cliques
    for _ in range(500):
        if time.time() > deadline - 20:
            break
        order = list(range(n))
        random.shuffle(order)
        order.sort(key=lambda i: (-degree[i] * 0.8 - random.random() * 0.2, moduli[i]))
        c = greedy_clique(order)
        if len(c) > len(best_clique) or (len(c) == len(best_clique) and c != best_clique):
            all_cliques.append(c)
            if len(c) > len(best_clique):
                best_clique = c
    
    # Deduplicate
    seen = set()
    unique_cliques = []
    for c in all_cliques:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_cliques.append(c)
    
    # Sort by size descending
    unique_cliques.sort(key=len, reverse=True)
    
    for clique in unique_cliques:
        if len(clique) <= best_size:
            break
        if time.time() > deadline - 10:
            break
        sol = solve_residues(clique, time_limit=10.0)
        if sol:
            if len(sol) > best_size:
                best_size = len(sol)
                best_solution = sol
    
    # If we couldn't find a solution for large cliques, try smaller ones
    # or use a different strategy
    
    # Alternative strategy: use the fact that if all moduli share a prime p,
    # we can assign residues based on their value mod p.
    # For all even moduli: assign r_i = i % 2 (alternating 0, 1).
    # But this only works if gcd(m_i, m_j) = 2 for all pairs, which is not true.
    
    # Let's try a specialized approach for the all-even subset:
    even_moduli = [i for i in range(n) if moduli[i] % 2 == 0]
    if len(even_moduli) > best_size:
        # For all-even moduli, gcd(m_i, m_j) >= 2.
        # If we assign r_i = 0 for all, then (0-0)%g = 0 for all g, which fails.
        # We need (r_j - r_i) % g != 0 for all pairs.
        # 
        # Idea: assign r_i = i mod 2. Then for pairs with gcd=2: (r_j - r_i) % 2 != 0 iff r_j != r_i mod 2.
        # But for pairs with gcd=4: (r_j - r_i) % 4 != 0. If r_i = 0, r_j = 1: 1%4 != 0. OK.
        # If r_i = 0, r_j = 2: but r_j = j%2, so r_j is 0 or 1. So r_j - r_i is 0 or 1.
        # 0%g = 0 for any g. So we can't have r_i = r_j for any pair.
        # 
        # So we need all residues to be distinct mod gcd(m_i, m_j) for each pair.
        # This is very restrictive.
        # 
        # Alternative: use r_i = i (the index in the sorted list).
        # Then r_j - r_i = j - i. We need (j - i) % gcd(m_i, m_j) != 0.
        # This might work for many cases.
        
        even_sorted = sorted(even_moduli, key=lambda i: moduli[i])
        k = len(even_sorted)
        assignment = list(range(k))  # r_i = i
        
        # Check if this works
        valid = True
        for a in range(k):
            for b in range(a + 1, k):
                g = gcd_table[even_sorted[a]][even_sorted[b]]
                if g > 1 and (assignment[b] - assignment[a]) % g == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            sol = [[assignment[i] % moduli[even_sorted[i]], moduli[even_sorted[i]]] for i in range(k)]
            if len(sol) > best_size:
                best_size = len(sol)
                best_solution = sol
        else:
            # Try different assignments
            # Use r_i = i * some_step
            for step in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
                assignment = [(i * step) for i in range(k)]
                valid = True
                for a in range(k):
                    for b in range(a + 1, k):
                        g = gcd_table[even_sorted[a]][even_sorted[b]]
                        if g > 1 and (assignment[b] - assignment[a]) % g == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    sol = [[assignment[i] % moduli[even_sorted[i]], moduli[even_sorted[i]]] for i in range(k)]
                    if len(sol) > best_size:
                        best_size = len(sol)
                        best_solution = sol
                    break
    
    # Try multiples of 3
    mult3 = [i for i in range(n) if moduli[i] % 3 == 0]
    if len(mult3) > best_size:
        mult3_sorted = sorted(mult3, key=lambda i: moduli[i])
        k = len(mult3_sorted)
        for step in [1, 2, 4, 5, 7, 8, 10, 11]:
            assignment = [(i * step) for i in range(k)]
            valid = True
            for a in range(k):
                for b in range(a + 1, k):
                    g = gcd_table[mult3_sorted[a]][mult3_sorted[b]]
                    if g > 1 and (assignment[b] - assignment[a]) % g == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                sol = [[assignment[i] % moduli[mult3_sorted[i]], moduli[mult3_sorted[i]]] for i in range(k)]
                if len(sol) > best_size:
                    best_size = len(sol)
                    best_solution = sol
                break
    
    # Try a combined approach: use backtracking on a carefully selected subset
    # that maximizes the chance of finding a valid residue assignment.
    
    # Select moduli that are all multiples of 2 and have small gcds with each other
    # (i.e., gcd is mostly 2, not larger).
    
    # For now, if we have a solution, return it.
    if best_solution:
        return best_solution
    
    # Fallback: simple greedy
    solution = []
    used = set()
    for idx in sorted(range(n), key=lambda i: (-degree[i], moduli[i])):
        m = moduli[idx]
        for r in range(m):
            ok = True
            for (r2, m2) in solution:
                g = _gcd(m, m2)
                if g > 1 and (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                solution.append([r, m])
                used.add(m)
                break
    
    if solution:
        return solution
    
    return [[0, 2]]
# EVOLVE-BLOCK-END
