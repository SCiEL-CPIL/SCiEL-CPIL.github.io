import math
import time
import random

MAX_MOD = 128

def gcd(a, b):
    return math.gcd(a, b)

def is_disjoint(r1, m1, r2, m2):
    g = gcd(m1, m2)
    if g == 1:
        return False
    return ((r2 - r1) % g) != 0

def verify_solution(pairs):
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        if m1 < 2 or m1 > MAX_MOD or r1 < 0 or r1 >= m1:
            return False
        for j in range(i + 1, n):
            r2, m2 = pairs[j]
            if m1 == m2:
                return False
            if not is_disjoint(r1, m1, r2, m2):
                return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    
    # Strategy: Model as maximum independent set on conflict graph
    # Nodes are (residue, modulus) pairs, edges denote disjointness violations
    # Use branch-and-bound with bitset-accelerated candidate filtering
    
    # First, let's use a smarter approach:
    # For each modulus, the residue determines which other moduli/residues are blocked
    # Key insight: if gcd(m1, m2) = g > 1, then we need (r2 - r1) % g != 0
    # This means for each pair of moduli, there are g-1 "bad" residue differences out of g
    
    # Let's try a different approach: use all moduli and find compatible residues
    # via constraint propagation
    
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd for all pairs
    n_mods = len(all_moduli)
    gcd_table = {}
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            m1, m2 = all_moduli[i], all_moduli[j]
            g = gcd(m1, m2)
            if g > 1:
                gcd_table[(i, j)] = g
                gcd_table[(j, i)] = g
    
    # Use backtracking with constraint propagation
    # Order moduli by decreasing "constraint" - those with more shared factors first
    def constraint_count(m_idx):
        m = all_moduli[m_idx]
        count = 0
        for j in range(n_mods):
            if j != m_idx:
                g = gcd(m, all_moduli[j])
                if g > 1:
                    count += g
        return count
    
    mod_order = sorted(range(n_mods), key=lambda i: -constraint_count(i))
    
    best_solution = []
    
    def backtrack(idx, assigned, residues_used):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        # Update best if we have more
        if len(assigned) > len(best_solution):
            best_solution = list(assigned)
        
        if idx == len(mod_order):
            return
        
        m_idx = mod_order[idx]
        m = all_moduli[m_idx]
        
        # Try to assign a residue
        # Use constraint propagation to filter candidates
        candidates = []
        for r in range(m):
            ok = True
            for (r2, m2_idx) in assigned:
                m2 = all_moduli[m2_idx]
                g = gcd(m, m2)
                if g > 1 and (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                candidates.append(r)
        
        # Try candidates in a smart order
        # Prefer residues that are "spread out"
        random.shuffle(candidates)
        
        for r in candidates:
            if time.time() > deadline:
                return
            assigned.append((r, m_idx))
            backtrack(idx + 1, assigned, residues_used)
            assigned.pop()
        
        # Option to skip this modulus
        backtrack(idx + 1, assigned, residues_used)
    
    backtrack(0, [], set())
    
    if best_solution and verify_solution(best_solution):
        # Convert back to [residue, modulus] format
        result = [[r, all_moduli[m_idx]] for (r, m_idx) in best_solution]
        return result
    
    # Fallback: greedy approach with even moduli
    even_moduli = list(range(2, MAX_MOD + 1, 2))
    pairs = []
    for m in sorted(even_moduli, reverse=True):
        for r in range(m):
            ok = True
            for (r2, m2) in pairs:
                if not is_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                pairs.append([r, m])
                break
    
    if pairs and verify_solution(pairs):
        return pairs
    
    # Final fallback
    return [[m // 2, m] for m in [2, 4, 8, 16, 32, 64, 128]]
