import math
import time
import random
from functools import lru_cache
from itertools import combinations

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _lcm(a, b):
    return a // _gcd(a, b) * b

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect
    return (r2 - r1) % g != 0

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin for 90s limit
    
    # All moduli from 2 to 128
    all_moduli = list(range(2, 129))
    
    # Precompute gcd table for all pairs
    gcd_table = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = _gcd(m1, m2)
    
    # Precompute which moduli are "compatible" (gcd > 1)
    # For two moduli to potentially be disjoint, they must have gcd > 1
    compatible = {}
    for m in all_moduli:
        compatible[m] = set()
        for m2 in all_moduli:
            if m != m2 and _gcd(m, m2) > 1:
                compatible[m].add(m2)
    
    best_result = None
    best_count = 0
    
    # Strategy 1: Greedy with multiple random orderings
    # Focus on moduli that share factors with many others
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Create a random ordering, biased toward moduli with more compatible partners
        random.seed(attempt)
        
        # Weight by number of compatible moduli
        weights = [len(compatible[m]) + 1 for m in all_moduli]
        total_w = sum(weights)
        
        perm = []
        remaining = all_moduli[:]
        remaining_weights = weights[:]
        
        for _ in range(len(all_moduli)):
            if not remaining:
                break
            r = random.uniform(0, sum(remaining_weights))
            cumsum = 0
            for i, w in enumerate(remaining_weights):
                cumsum += w
                if r <= cumsum:
                    perm.append(remaining[i])
                    remaining.pop(i)
                    remaining_weights.pop(i)
                    break
        
        selected = []
        
        for m in perm:
            if time.time() > deadline:
                break
            
            # Check if m is coprime to any already selected modulus
            skip = False
            for (r, sm) in selected:
                if _gcd(m, sm) == 1:
                    skip = True
                    break
            if skip:
                continue
            
            # Try to find a residue for m that works
            # Use random order for residues to break symmetry
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                can_add = True
                for (sr, sm) in selected:
                    g = gcd_table.get((m, sm), 1)
                    if g == 1:
                        can_add = False
                        break
                    if (r - sr) % g == 0:
                        can_add = False
                        break
                if can_add:
                    selected.append((r, m))
                    found = True
                    break
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = selected[:]
        
        if best_count >= 50:
            break
    
    # Strategy 2: Structured approach using powers of 2
    # Powers of 2: 2, 4, 8, 16, 32, 64, 128
    # These can be made pairwise disjoint with appropriate residues
    
    structured = []
    for k in range(7):
        m = 1 << (k + 1)
        if m <= 128:
            r = (1 << k) - 1
            structured.append((r, m))
    
    # Verify
    valid = True
    for i in range(len(structured)):
        for j in range(i + 1, len(structured)):
            if not _disjoint(structured[i][0], structured[i][1], structured[j][0], structured[j][1]):
                valid = False
                break
        if not valid:
            break
    
    if valid and len(structured) > best_count:
        best_result = structured
        best_count = len(structured)
    
    # Strategy 3: Try to build a larger set using moduli that are multiples of small primes
    # Focus on even moduli, multiples of 3, etc.
    
    # Group moduli by their smallest prime factor
    spf = {}
    for m in all_moduli:
        for p in range(2, m + 1):
            if m % p == 0:
                spf[m] = p
                break
    
    # Try greedy within groups
    for prime in [2, 3, 5, 7]:
        if time.time() > deadline:
            break
        
        group = [m for m in all_moduli if spf.get(m) == prime]
        if len(group) < 2:
            continue
        
        random.seed(1000 + prime)
        perm = group[:]
        random.shuffle(perm)
        
        selected = []
        for m in perm:
            # Check compatibility with existing
            skip = False
            for (r, sm) in selected:
                if _gcd(m, sm) == 1:
                    skip = True
                    break
            if skip:
                continue
            
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                can_add = True
                for (sr, sm) in selected:
                    g = gcd_table.get((m, sm), 1)
                    if g == 1 or (r - sr) % g == 0:
                        can_add = False
                        break
                if can_add:
                    selected.append((r, m))
                    found = True
                    break
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = selected[:]
    
    # Strategy 4: Backtracking with pruning for small subsets
    # Try to find a large clique in the compatibility graph
    # where we can assign residues
    
    # Build compatibility graph
    compat_graph = {}
    for m in all_moduli:
        compat_graph[m] = set()
        for m2 in all_moduli:
            if m != m2 and _gcd(m, m2) > 1:
                compat_graph[m].add(m2)
    
    # Try to find large cliques and assign residues
    # This is expensive, so limit to small candidate sets
    
    # Candidate set: moduli that are compatible with many others
    candidates = sorted(all_moduli, key=lambda m: len(compat_graph[m]), reverse=True)[:30]
    
    # Try greedy clique finding on candidates
    random.seed(42)
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        perm = candidates[:]
        random.shuffle(perm)
        
        selected = []
        for m in perm:
            # Check if m is compatible with all selected
            skip = False
            for (r, sm) in selected:
                if _gcd(m, sm) == 1:
                    skip = True
                    break
            if skip:
                continue
            
            # Try residues
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                can_add = True
                for (sr, sm) in selected:
                    g = gcd_table.get((m, sm), 1)
                    if g == 1 or (r - sr) % g == 0:
                        can_add = False
                        break
                if can_add:
                    selected.append((r, m))
                    found = True
                    break
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = selected[:]
    
    # Strategy 5: Try to extend the best result by swapping residues
    if best_result and best_count < 60:
        if time.time() < deadline - 5:
            # Try local search: swap residues to enable more additions
            current = best_result[:]
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        # Try swapping residues
                        r_i, m_i = current[i]
                        r_j, m_j = current[j]
                        
                        # Try all pairs of residues
                        for new_r_i in range(m_i):
                            for new_r_j in range(m_j):
                                # Check if swap is valid
                                valid_swap = True
                                for k in range(len(current)):
                                    if k == i or k == j:
                                        continue
                                    r_k, m_k = current[k]
                                    if _gcd(m_i, m_k) > 1 and (new_r_i - r_k) % _gcd(m_i, m_k) == 0:
                                        valid_swap = False
                                        break
                                    if _gcd(m_j, m_k) > 1 and (new_r_j - r_k) % _gcd(m_j, m_k) == 0:
                                        valid_swap = False
                                        break
                                if not valid_swap:
                                    continue
                                
                                # Check if this enables adding a new modulus
                                new_current = current[:]
                                new_current[i] = (new_r_i, m_i)
                                new_current[j] = (new_r_j, m_j)
                                
                                for m_new in all_moduli:
                                    if any(m_new == sm for (sr, sm) in new_current):
                                        continue
                                    
                                    # Check compatibility
                                    compat = True
                                    for (sr, sm) in new_current:
                                        if _gcd(m_new, sm) == 1:
                                            compat = False
                                            break
                                    if not compat:
                                        continue
                                    
                                    # Try to find a residue
                                    for r_new in range(m_new):
                                        can_add = True
                                        for (sr, sm) in new_current:
                                            g = gcd_table.get((m_new, sm), 1)
                                            if g == 1 or (r_new - sr) % g == 0:
                                                can_add = False
                                                break
                                        if can_add:
                                            new_current.append((r_new, m_new))
                                            if len(new_current) > best_count:
                                                best_count = len(new_current)
                                                best_result = new_current[:]
                                                current = new_current[:]
                                                improved = True
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
    
    # Final fallback
    if not best_result:
        best_result = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    
    return [[r, m] for r, m in best_result]
