"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys
from functools import lru_cache

MAX_MOD = 128
DEADLINE_MARGIN = 5.0  # seconds before the 90s kill

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _valid(pairs):
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if not _disjoint(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]):
                return False
    return True

def _compute_conflicts(all_mods):
    """Precompute for each pair of moduli, the gcd and conflict structure."""
    n = len(all_mods)
    mod_index = {m: i for i, m in enumerate(all_mods)}
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcds[i][j] = math.gcd(all_mods[i], all_mods[j])
            gcds[j][i] = gcds[i][j]
    return gcds, mod_index

def _backtrack_search(all_mods, time_limit, max_depth=None):
    """
    Backtracking with forward checking.
    We process moduli in a fixed order and try to assign residues.
    State: for each unassigned modulus, track which residues are still possible.
    """
    n = len(all_mods)
    if max_depth is None:
        max_depth = n
    
    # Sort moduli by a heuristic: moduli that conflict with many others first
    # Use degree-based ordering: count how many other moduli share a gcd > 1
    degrees = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(all_mods[i], all_mods[j]) > 1:
                degrees[i] += 1
                degrees[j] += 1
    
    # Order: higher degree first (more constrained)
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # For backtracking, we'll maintain:
    # - assigned: dict from modulus_index to residue
    # - For each unassigned modulus, the set of valid residues given current assignments
    
    assigned = {}
    # domains[i] = set of residues that are valid for modulus all_mods[i] given current assignments
    domains = [set(range(all_mods[i])) for i in range(n)]
    
    # We process in the order given by 'order'
    # But we need to track which positions in 'order' have been assigned
    
    best_solution = []
    best_len = 0
    
    def check_validity(assigned_dict):
        """Check if the current assignment is valid (all pairs disjoint)."""
        mods_list = list(assigned_dict.items())
        for i in range(len(mods_list)):
            for j in range(i+1, len(mods_list)):
                r1, m1 = mods_list[i]
                r2, m2 = mods_list[j]
                if not _disjoint(r1, m1, r2, m2):
                    return False
        return True
    
    def backtrack(pos, current_assigned, current_domains):
        """
        pos: position in the 'order' list
        current_assigned: dict from modulus index to residue
        current_domains: list of sets of valid residues for each modulus index
        """
        nonlocal best_solution, best_len
        
        if time.time() > time_limit:
            return
        
        # If we've processed all moduli in order
        if pos >= n:
            # Update best
            sol = [(current_assigned[i], all_mods[i]) for i in sorted(current_assigned.keys())]
            if len(sol) > best_len:
                best_len = len(sol)
                best_solution = sol
            return
        
        # Try to assign a residue to order[pos]
        m_idx = order[pos]
        m = all_mods[m_idx]
        
        # Get current domain for this modulus
        domain = list(current_domains[m_idx])
        random.shuffle(domain)  # Randomize to explore different branches
        
        for r in domain:
            if time.time() > time_limit:
                return
            
            # Check if r is compatible with all already assigned moduli
            compatible = True
            new_domains = [set(d) for d in current_domains]
            
            for other_idx, other_r in current_assigned.items():
                other_m = all_mods[other_idx]
                if not _disjoint(r, m, other_r, other_m):
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Forward checking: update domains for unassigned moduli
            valid_branch = True
            for unassigned_idx in range(n):
                if unassigned_idx == m_idx or unassigned_idx in current_assigned:
                    continue
                # Check if any residue in domain is compatible with (r, m)
                any_compatible = False
                for ur in current_domains[unassigned_idx]:
                    if _disjoint(r, m, ur, all_mods[unassigned_idx]):
                        any_compatible = True
                        break
                if not any_compatible:
                    valid_branch = False
                    break
            
            if not valid_branch:
                continue
            
            # Recurse
            new_assigned = dict(current_assigned)
            new_assigned[m_idx] = r
            
            # Only recurse if we haven't exceeded max_depth or if we're still improving
            if len(new_assigned) <= max_depth:
                backtrack(pos + 1, new_assigned, current_domains)
    
    # Start backtracking
    backtrack(0, {}, domains)
    
    return best_solution

def _greedy_build(moduli_order, time_limit, randomize_residues=False):
    """Given an ordered list of moduli, greedily assign residues."""
    pairs = []
    used_mods = set()
    for m in moduli_order:
        if m in used_mods:
            continue
        if time.time() > time_limit:
            break
        best_r = None
        residues = list(range(m))
        if randomize_residues:
            random.shuffle(residues)
        
        for r in residues:
            ok = True
            for (r2, m2) in pairs:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                best_r = r
                break
        
        if best_r is not None:
            pairs.append((best_r, m))
            used_mods.add(m)
    return pairs

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN  # stay well under 90s

    all_mods = list(range(2, MAX_MOD + 1))
    
    best = []
    best_len = 0

    # Strategy 1: Backtracking with forward checking
    # Use a moderate max_depth to avoid exponential blowup
    # Try different max_depth values
    for max_depth in [20, 25, 30]:
        if time.time() > deadline:
            break
        sol = _backtrack_search(all_mods, deadline, max_depth)
        if len(sol) > best_len:
            best = sol
            best_len = len(sol)
    
    # Strategy 2: Greedy with different orderings as fallback/improvement
    orderings = [
        all_mods,
        list(reversed(all_mods)),
        sorted(all_mods, key=lambda m: -m.bit_count()),
        sorted(all_mods, key=lambda m: m.bit_count()),
    ]
    
    for _ in range(5):
        shuffled = all_mods[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    for order in orderings:
        if time.time() > deadline:
            break
        for rand_res in [False, True]:
            if time.time() > deadline:
                break
            pairs = _greedy_build(order, deadline, randomize_residues=rand_res)
            if len(pairs) > best_len:
                best = pairs
                best_len = len(pairs)
                
    # Strategy 3: Local search to improve
    current_best = [[r, m] for r, m in best]
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        perturbed = current_best[:]
        if perturbed:
            k = min(len(perturbed), random.randint(1, 3))
            indices = random.sample(range(len(perturbed)), k)
            indices.sort(reverse=True)
            for idx in indices:
                perturbed.pop(idx)
        
        used_mods = set(m for _, m in perturbed)
        remaining_mods = [m for m in all_mods if m not in used_mods]
        remaining_mods.sort()
        
        new_pairs = perturbed[:]
        for m in remaining_mods:
            if time.time() > deadline:
                break
            found = False
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                ok = True
                for (r2, m2) in new_pairs:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    new_pairs.append([r, m])
                    found = True
                    break
        
        if len(new_pairs) > len(current_best):
            current_best = new_pairs
            best_len = len(new_pairs)
            best = [[r, m] for r, m in current_best]
            
    return best
# EVOLVE-BLOCK-END
