import math
import time
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    moduli = list(range(2, 129))
    n = len(moduli)
    
    # Precompute GCDs
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = _gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    # Precompute prime factors for each modulus
    def prime_factors(m):
        facs = set()
        d = 2
        while d * d <= m:
            if m % d == 0:
                facs.add(d)
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            facs.add(m)
        return facs
    
    pf_cache = {m: prime_factors(m) for m in moduli}
    
    # Group moduli by their prime factor families
    # For each prime p, find all moduli divisible by p
    prime_families = {}
    for i, m in enumerate(moduli):
        for p in pf_cache[m]:
            if p not in prime_families:
                prime_families[p] = []
            prime_families[p].append(i)
    
    # Sort families by size (largest first)
    family_order = sorted(prime_families.keys(), key=lambda p: -len(prime_families[p]))
    
    best_count = 0
    best_result = []
    
    def is_valid(i, r, selected):
        """Check if residue r for modulus index i is disjoint from all in selected."""
        for (j, rj) in selected:
            g = gcds[i][j]
            if g > 1 and (r - rj) % g == 0:
                return False
        return True
    
    def find_valid_residue(i, selected):
        """Find a valid residue for modulus i given selected classes."""
        m = moduli[i]
        for r in range(m):
            if is_valid(i, r, selected):
                return r
        return None
    
    def greedy_solve(order):
        """Greedy: process moduli in given order, pick first valid residue."""
        selected = []
        for i in order:
            m = moduli[i]
            r = find_valid_residue(i, selected)
            if r is not None:
                selected.append((i, r))
        return selected
    
    def local_search(selected, time_budget=5.0):
        """Improve by trying to add missing moduli with different residues, or swapping."""
        selected = list(selected)
        end_time = time.time() + time_budget
        improved = True
        
        while improved and time.time() < end_time:
            improved = False
            in_set = set(i for i, _ in selected)
            
            # Try to add a missing modulus
            for i in range(n):
                if i in in_set:
                    continue
                m = moduli[i]
                r = find_valid_residue(i, selected)
                if r is not None:
                    selected.append((i, r))
                    improved = True
                    break
            
            # Try to swap residues to enable adding more
            if not improved and time.time() < end_time:
                for idx in range(len(selected)):
                    i, r = selected[idx]
                    m = moduli[i]
                    found_swap = False
                    
                    for r_new in range(m):
                        if r_new == r:
                            continue
                        
                        # Check if r_new is valid against all others
                        valid = True
                        for j, (k, rk) in enumerate(selected):
                            if j == idx:
                                continue
                            g = gcds[i][k]
                            if g > 1 and (r_new - rk) % g == 0:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Try swapping and see if we can add a new modulus
                        old_r = selected[idx][1]
                        selected[idx] = (i, r_new)
                        
                        in_set_after_swap = set(i for i, _ in selected)
                        for i2 in range(n):
                            if i2 in in_set_after_swap:
                                continue
                            m2 = moduli[i2]
                            r2 = find_valid_residue(i2, selected)
                            if r2 is not None:
                                selected.append((i2, r2))
                                found_swap = True
                                break
                        
                        if not found_swap:
                            selected[idx] = (i, old_r)
                    
                    if found_swap:
                        improved = True
                        break
        
        return selected
    
    def clique_based_search(family_prime, time_budget=10.0):
        """Search within a prime factor family using clique approach."""
        family_indices = prime_families[family_prime]
        end_time = time.time() + time_budget
        
        # For moduli in the same prime family, the constraint is simpler
        # Build a subproblem focusing on these moduli
        
        selected = []
        used = set()
        
        # First, try to build a large set within this family
        # Sort by modulus (smaller first for more constraints)
        sorted_indices = sorted(family_indices, key=lambda i: moduli[i])
        
        for i in sorted_indices:
            if time.time() > end_time:
                break
            r = find_valid_residue(i, selected)
            if r is not None:
                selected.append((i, r))
                used.add(i)
        
        # Then try to add moduli from outside the family
        for i in range(n):
            if time.time() > end_time:
                break
            if i in used:
                continue
            r = find_valid_residue(i, selected)
            if r is not None:
                selected.append((i, r))
                used.add(i)
        
        # Polish with local search
        remaining_time = max(0.5, end_time - time.time())
        selected = local_search(selected, time_budget=remaining_time)
        
        return selected
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Strategy 1: Clique-based search on largest prime family
        if attempts % 4 == 1 and family_order:
            p = family_order[0]  # Largest family (usually 2)
            selected = clique_based_search(p, time_budget=min(15.0, max(1.0, deadline - time.time() - 5.0)))
        
        # Strategy 2: Random order greedy
        elif attempts % 4 == 2:
            indices = list(range(n))
            random.shuffle(indices)
            selected = greedy_solve(indices)
            if len(selected) > best_count or (best_count > 0 and len(selected) >= best_count):
                remaining = max(0.5, deadline - time.time() - 2.0)
                selected = local_search(selected, time_budget=min(5.0, remaining))
        
        # Strategy 3: Most constrained first (smaller moduli)
        elif attempts % 4 == 3:
            indices = list(range(n))
            indices.sort(key=lambda i: moduli[i])
            selected = greedy_solve(indices)
            if len(selected) > best_count or (best_count > 0 and len(selected) >= best_count):
                remaining = max(0.5, deadline - time.time() - 2.0)
                selected = local_search(selected, time_budget=min(5.0, remaining))
        
        # Strategy 4: Reverse order (largest first)
        else:
            indices = list(range(n))
            indices.sort(key=lambda i: moduli[i], reverse=True)
            selected = greedy_solve(indices)
            if len(selected) > best_count or (best_count > 0 and len(selected) >= best_count):
                remaining = max(0.5, deadline - time.time() - 2.0)
                selected = local_search(selected, time_budget=min(5.0, remaining))
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = [[r, moduli[i]] for i, r in selected]
            
            # If we've found a very large set, try to polish more
            if best_count > 100:
                for _ in range(3):
                    if time.time() > deadline - 3:
                        break
                    indices = list(range(n))
                    random.shuffle(indices)
                    selected = greedy_solve(indices)
                    remaining = max(0.5, deadline - time.time() - 2.0)
                    selected_polished = local_search(selected, time_budget=min(3.0, remaining))
                    if len(selected_polished) > best_count:
                        best_count = len(selected_polished)
                        best_result = [[r, moduli[i]] for i, r in selected_polished]
                break
    
    if not best_result:
        # Fallback
        return [[1, 2]]
        
    return best_result
