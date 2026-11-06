"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect
    return (r2 - r1) % g != 0

def _can_add(r_new, m_new, existing):
    """Check if new class is disjoint from all existing."""
    for (r, m) in existing:
        if not _disjoint(r_new, m_new, r, m):
            return False
    return True

def _count_conflicts(r_new, m_new, candidate_moduli, existing):
    """Count how many potential future moduli would be blocked by choosing this residue."""
    count = 0
    for m in candidate_moduli:
        if m == m_new:
            continue
        g = _gcd(m_new, m)
        if g > 1:
            # Check if this residue would block all residues for m
            # Actually, it blocks residues r' where (r' - r_new) % g == 0
            # Number of such residues in [0, m) is m // g (approximately)
            # But we care about whether ANY residue for m would still be available
            # For now, just count if gcd > 1 as a soft penalty
            count += 1
    return count

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin before 90s kill
    
    # Candidate moduli from 2 to 128
    all_moduli = list(range(2, 129))
    
    best_solution = []
    best_size = 0
    
    # Strategy: Use a greedy approach with multiple orderings.
    # Key insight: moduli that share many factors with other moduli are "harder" 
    # to place, so we should place them first (most-constrained-first).
    # Alternatively, moduli that are coprime to many others are "easier" and 
    # can be placed later.
    
    # Precompute gcd relationships
    n_moduli = len(all_moduli)
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            g = _gcd(all_moduli[i], all_moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Score each modulus by how "constrained" it is:
    # A modulus is more constrained if it shares factors with many other moduli
    # because choosing a residue for it will block more future options.
    def constraint_score(m_idx):
        m = all_moduli[m_idx]
        shared = 0
        for j in range(n_moduli):
            if j != m_idx and gcd_table[m_idx][j] > 1:
                shared += 1
        return shared
    
    # Most constrained first
    moduli_by_constraint = sorted(range(n_moduli), key=lambda i: -constraint_score(i))
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Most constrained first
    strategies.append(moduli_by_constraint[:])
    
    # Strategy 2: Least constrained first
    strategies.append(sorted(range(n_moduli), key=lambda i: constraint_score(i)))
    
    # Strategy 3: Random orderings
    for _ in range(10):
        order = list(range(n_moduli))
        random.shuffle(order)
        strategies.append(order)
    
    # Strategy 4: Group by prime factors - moduli with same prime factors first
    # Moduli divisible by 2 first, then by 3, etc.
    def prime_factor_count(m):
        return sum(1 for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127] if m % p == 0)
    
    strategies.append(sorted(range(n_moduli), key=lambda i: -prime_factor_count(all_moduli[i])))
    strategies.append(sorted(range(n_moduli), key=lambda i: prime_factor_count(all_moduli[i])))
    
    # For each strategy, try multiple residue selection heuristics
    for strat_idx, moduli_order in enumerate(strategies):
        if time.time() > deadline:
            break
        
        # Try a few different residue selection methods for each ordering
        for residue_strategy in range(3):  # 0=random, 1=first valid, 2=least conflict
            if time.time() > deadline:
                break
            
            current = []
            used_moduli = set()
            
            for m_idx in moduli_order:
                if time.time() > deadline:
                    break
                
                m = all_moduli[m_idx]
                if m in used_moduli:
                    continue
                
                # Try to find a valid residue
                residues = list(range(m))
                
                if residue_strategy == 0:
                    random.shuffle(residues)
                    found = False
                    for r in residues:
                        if _can_add(r, m, current):
                            current.append([r, m])
                            used_moduli.add(m)
                            found = True
                            break
                elif residue_strategy == 1:
                    found = False
                    for r in residues:
                        if _can_add(r, m, current):
                            current.append([r, m])
                            used_moduli.add(m)
                            found = True
                            break
                else:  # residue_strategy == 2: pick residue that minimizes future conflicts
                    valid_residues = [r for r in residues if _can_add(r, m, current)]
                    if valid_residues:
                        # Score each valid residue by how many future moduli it blocks
                        remaining_moduli = [all_moduli[j] for j in moduli_order if all_moduli[j] not in used_moduli and all_moduli[j] != m]
                        best_r = min(valid_residues, key=lambda r: sum(1 for m2 in remaining_moduli if _gcd(m, m2) > 1 and (r - 0) % _gcd(m, m2) == 0))
                        current.append([best_r, m])
                        used_moduli.add(m)
                    # If no valid residue, skip this modulus
            
            if len(current) > best_size:
                best_size = len(current)
                best_solution = [x[:] for x in current]
    
    # Local improvement: try to swap residues or add/remove moduli
    if best_solution and time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            sol = [x[:] for x in best_solution]
            used = set(m for _, m in sol)
            
            # Try adding a missing modulus
            for m in all_moduli:
                if time.time() > deadline - 10:
                    break
                if m in used:
                    continue
                for r in range(m):
                    if _can_add(r, m, sol):
                        sol.append([r, m])
                        used.add(m)
                        improved = True
                        if len(sol) > best_size:
                            best_size = len(sol)
                            best_solution = [x[:] for x in sol]
                        break
                if improved:
                    break
            
            # Try swapping a residue to free up a modulus
            if not improved:
                for i, (r1, m1) in enumerate(sol):
                    if time.time() > deadline - 10:
                        break
                    for r_new in range(m1):
                        if r_new == r1:
                            continue
                        # Check if new residue is valid with all other classes
                        sol[i] = [r_new, m1]
                        if all(_disjoint(r_new, m1, r2, m2) for j, (r2, m2) in enumerate(sol) if j != i):
                            # Try to add a missing modulus
                            for m_add in all_moduli:
                                if m_add in used:
                                    continue
                                for r_add in range(m_add):
                                    if _can_add(r_add, m_add, sol):
                                        sol.append([r_add, m_add])
                                        used.add(m_add)
                                        improved = True
                                        if len(sol) > best_size:
                                            best_size = len(sol)
                                            best_solution = [x[:] for x in sol]
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    if improved:
                        break
                    else:
                        sol[i] = [r1, m1]
    
    if not best_solution:
        return [[1, 2]]
        
    return best_solution
# EVOLVE-BLOCK-END
