"""Strategy: Deterministic greedy with lookahead + local search refinement.
Key insight: For each modulus m, the number of residues r that conflict with an existing
set S is at most sum over (m', r') in S of gcd(m, m') / m' ... actually it's the union
of congruence classes mod gcd(m, m'). The number of conflicting residues is at most
sum_{(m',r') in S} gcd(m, m') but these may overlap.

A better approach: For a given modulus m and existing set S, the forbidden residues are
those r where exists (m', r') in S with r ≡ r' (mod gcd(m, m')).
The number of such forbidden residues is at most sum_{(m',r') in S} 1 (each forbids a
congruence class mod g, which has m/g elements in [0, m)). So forbidden count <= sum m/gcd(m,m').

If sum m/gcd(m, m') < m, then there's always a valid residue.
"""
import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    moduli = list(range(20, 201))
    
    # Precompute gcd table
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def find_valid_residue(m, existing_set):
        """Find a residue r in [0, m) that doesn't conflict with existing_set.
        Returns r or None."""
        # Collect all forbidden residues
        forbidden = set()
        for m2, r2 in existing_set:
            g = gcd(m, m2)
            # r must satisfy r ≡ r2 (mod g) is FORBIDDEN
            # So forbidden residues are r2, r2+g, r2+2g, ... < m
            for r in range(r2 % g, m, g):
                forbidden.add(r)
        
        valid = [r for r in range(m) if r not in forbidden]
        return valid

    def score_set(classes):
        """Calculate score as Fraction."""
        if not classes:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for m, r in classes:
            s += Fraction(1, m)
        return s

    def greedy_with_lookahead():
        """Greedy: process moduli in order of increasing m (highest 1/m first).
        For each modulus, pick a residue that minimizes the number of future conflicts."""
        current_set = []
        
        for m in sorted(moduli):  # ascending m = descending 1/m
            valid_residues = find_valid_residue(m, current_set)
            if not valid_residues:
                continue
            
            # Choose residue that minimizes impact on future moduli
            # Heuristic: prefer residues that are "spread out" - minimize conflicts
            # with moduli that haven't been processed yet
            best_r = None
            best_cost = float('inf')
            
            # Limit evaluation for speed
            if len(valid_residues) > 20:
                # Sample
                sample = random.sample(valid_residues, 20)
            else:
                sample = valid_residues
            
            for r in sample:
                cost = 0
                # Estimate conflicts with future moduli
                for m2 in moduli:
                    if m2 <= m:  # already processed or same
                        continue
                    g = gcd(m, m2)
                    # This residue r creates a conflict class: x ≡ r (mod g) in [0, m2)
                    # Number of m2-residues that would be blocked: m2 // g
                    # But this is an overestimate; just use as proxy
                    cost += 1.0 / (m2 // g + 1)
                if cost < best_cost:
                    best_cost = cost
                    best_r = r
            
            current_set.append([m, best_r])
        
        return current_set

    def local_search(initial_set):
        """Improve via: 1) add missing moduli, 2) swap to better moduli, 3) change residues."""
        current_set = [x[:] for x in initial_set]
        current_score = score_set(current_set)
        current_mods = set(m for m, r in current_set)
        candidates = [m for m in moduli if m not in current_mods]
        
        improved = True
        max_iters = 100
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # 1. Try adding a candidate
            for m_new in sorted(candidates):  # smallest m first (highest weight)
                valid_r = find_valid_residue(m_new, current_set)
                if valid_r:
                    # Pick a good residue
                    r = random.choice(valid_r[:min(len(valid_r), 10)]) if len(valid_r) > 10 else valid_r[0]
                    current_set.append([m_new, r])
                    current_score += Fraction(1, m_new)
                    current_mods.add(m_new)
                    candidates.remove(m_new)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swapping
            for i, (m_old, r_old) in enumerate(current_set):
                swap_candidates = [m for m in candidates if m < m_old]
                if not swap_candidates:
                    continue
                for m_new in swap_candidates[:10]:  # limit
                    temp_set = current_set[:i] + current_set[i+1:]
                    valid_r = find_valid_residue(m_new, temp_set)
                    if valid_r:
                        r_new = random.choice(valid_r[:min(len(valid_r), 10)]) if len(valid_r) > 10 else valid_r[0]
                        delta_score = Fraction(1, m_new) - Fraction(1, m_old)
                        if delta_score > 0:
                            current_set[i] = [m_new, r_new]
                            current_score += delta_score
                            current_mods.remove(m_old)
                            current_mods.add(m_new)
                            candidates.remove(m_new)
                            candidates.append(m_old)
                            improved = True
                            break
                if improved:
                    break
        
        return current_set

    def try_all_residues_greedy(seed=0):
        """Deterministic greedy trying all residues, picking the one that leaves most options."""
        rng = random.Random(seed)
        current_set = []
        
        for m in sorted(moduli):
            valid_residues = find_valid_residue(m, current_set)
            if not valid_residues:
                continue
            
            # Score each residue by how many future moduli it doesn't block
            best_r = valid_residues[0]
            best_score = -1
            
            if len(valid_residues) <= 50:
                for r in valid_residues:
                    sc = 0
                    for m2 in moduli:
                        if m2 <= m:
                            continue
                        g = gcd(m, m2)
                        # Count how many residues of m2 are blocked
                        blocked = m2 // g
                        sc -= blocked / m2  # minimize blocking
                    if sc > best_score:
                        best_score = sc
                        best_r = r
            else:
                # Sample
                for r in rng.sample(valid_residues, 50):
                    sc = 0
                    for m2 in moduli:
                        if m2 <= m:
                            continue
                        g = gcd(m, m2)
                        blocked = m2 // g
                        sc -= blocked / m2
                    if sc > best_score:
                        best_score = sc
                        best_r = r
            
            current_set.append([m, best_r])
        
        return current_set

    # Run multiple strategies
    best_set = []
    best_score = Fraction(0, 1)

    # Strategy 1: Simple greedy with random restarts
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        rng = random.Random(attempt * 7919)
        current_set = []
        for m in sorted(moduli):
            valid_r = find_valid_residue(m, current_set)
            if not valid_r:
                continue
            r = rng.choice(valid_r)
            current_set.append([m, r])
        s = score_set(current_set)
        if s > best_score:
            best_score = s
            best_set = [x[:] for x in current_set]

    # Strategy 2: Lookahead greedy
    if time.time() < deadline - 30:
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            s_set = try_all_residues_greedy(seed)
            s = score_set(s_set)
            if s > best_score:
                best_score = s
                best_set = [x[:] for x in s_set]

    # Strategy 3: Local search refinement on best found so far
    if time.time() < deadline - 10 and best_set:
        refined = local_search(best_set)
        s_ref = score_set(refined)
        if s_ref > best_score:
            best_score = s_ref
            best_set = refined

    # Strategy 4: More local search from different starting points
    if time.time() < deadline - 5:
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            rng = random.Random(attempt * 104729)
            current_set = []
            for m in sorted(moduli):
                valid_r = find_valid_residue(m, current_set)
                if not valid_r:
                    continue
                r = rng.choice(valid_r)
                current_set.append([m, r])
            refined = local_search(current_set)
            s_ref = score_set(refined)
            if s_ref > best_score:
                best_score = s_ref
                best_set = refined

    return best_set
