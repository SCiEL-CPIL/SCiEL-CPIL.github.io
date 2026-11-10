"""Maximize pairwise-disjoint residue classes with distinct moduli 2..128 using local search."""
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

def _random_valid_clique():
    """Generate a random valid clique by random moduli selection and greedy residue assignment."""
    # Start with a random subset of moduli, try to assign residues
    # To ensure validity, we pick moduli that are likely to work (e.g., high overlap in prime factors)
    # Or just try random sets and validate.
    
    # Strategy: Pick a random number of moduli (start small, grow)
    # But we want to maximize, so let's try to build a large one.
    # A robust way: Start with all moduli, randomly remove until valid? No, that's slow.
    # Better: Greedy construction with randomization.
    
    moduli_pool = list(range(2, MAX_MOD + 1))
    random.shuffle(moduli_pool)
    
    classes = []
    for m in moduli_pool:
        # Try to find a residue r for m that is disjoint from all existing classes
        found = False
        # Try random residues first, then fallback to linear scan if needed (or just linear)
        # Since m <= 128, linear scan is fast.
        for r in range(m):
            ok = True
            for r2, m2 in classes:
                g = gcd(m, m2)
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                classes.append([r, m])
                found = True
                break
        # If no residue found for this m, skip it (we don't add it)
        
    if not classes:
        # Fallback: at least one class
        classes = [[0, 2]]
    return classes

def _local_search(cliques, t0, deadline):
    """
    Perform local search on a list of cliques.
    Moves:
    1. Swap modulus in a class with an unused modulus (if valid residue exists).
    2. Change residue of a class (if still valid).
    3. Add a new modulus (if valid residue exists).
    4. Remove a modulus (to allow adding others? or just to escape local optima).
    """
    best_clique = max(cliques, key=len)
    best_len = len(best_clique)
    
    # Precompute gcds for all pairs of moduli 2..128 to speed up checks
    # gcd_cache[m1][m2]
    gcd_cache = [[0]*(MAX_MOD+1) for _ in range(MAX_MOD+1)]
    for m1 in range(2, MAX_MOD+1):
        for m2 in range(m1, MAX_MOD+1):
            g = gcd(m1, m2)
            gcd_cache[m1][m2] = g
            gcd_cache[m2][m1] = g
            
    def is_valid_state(clique):
        # clique is list of [r, m]
        # Check distinct moduli (assumed in our representation if we track used_mods)
        # Check disjointness
        n = len(clique)
        for i in range(n):
            r1, m1 = clique[i]
            for j in range(i+1, n):
                r2, m2 = clique[j]
                g = gcd_cache[m1][m2]
                if (r1 - r2) % g == 0:
                    return False
        return True

    # We maintain a set of used moduli for quick lookup
    used_mods = set(m for r, m in best_clique)
    
    # Current state
    current = [c[:] for c in best_clique]
    current_used = set(used_mods)
    
    # Precompute available residues for each modulus given current clique?
    # That's expensive. Let's just check on the fly.
    
    # Move 1: Add a new modulus
    # Move 2: Change residue of existing
    # Move 3: Swap modulus
    
    # Simulated Annealing / Randomized Hill Climbing
    # Since the objective is just "maximize count", we accept moves that increase count.
    # We also accept moves that keep count same (to explore) or decrease count (with low prob) to escape local optima.
    
    # Let's stick to a simple randomized hill climbing with restarts if stuck.
    # Actually, "Add" and "Swap" are the main drivers for increasing size.
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    while time.time() < deadline:
        moved = False
        # Try random moves
        for _ in range(100): # Limit moves per iteration to avoid hanging on one bad path
            if time.time() > deadline:
                break
            
            move_type = random.choice(['add', 'swap', 'residue'])
            
            if move_type == 'add':
                # Pick a random unused modulus
                unused = [m for m in all_mods if m not in current_used]
                if not unused:
                    continue
                m_new = random.choice(unused)
                # Find a valid residue for m_new
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
                    new_used = current_used | {m_new}
                    if is_valid_state(new_clique): # Should be true by construction, but double check
                        current = new_clique
                        current_used = new_used
                        moved = True
                        if len(current) > best_len:
                            best_len = len(current)
                            best_clique = [c[:] for c in current]
                            
            elif move_type == 'swap':
                # Pick an existing index and a new modulus
                i = random.randint(0, len(current)-1)
                old_m = current[i][1]
                unused = [m for m in all_mods if m not in current_used]
                if not unused:
                    continue
                m_new = random.choice(unused)
                # Find valid residue for m_new against (current without i)
                valid_res = []
                for r in range(m_new):
                    ok = True
                    for j, (r2, m2) in enumerate(current):
                        if j == i:
                            continue
                        g = gcd_cache[m_new][m2]
                        if (r - r2) % g == 0:
                            ok = False
                            break
                    if ok:
                        valid_res.append(r)
                if valid_res:
                    r_new = random.choice(valid_res)
                    # Construct new clique
                    new_clique = [c[:] for c in current]
                    new_clique[i] = [r_new, m_new]
                    new_used = set(current_used)
                    new_used.remove(old_m)
                    new_used.add(m_new)
                    if is_valid_state(new_clique):
                        current = new_clique
                        current_used = new_used
                        moved = True
                        # Size doesn't change, so best_len doesn't update, but we explore.
                        
            elif move_type == 'residue':
                i = random.randint(0, len(current)-1)
                m_cur = current[i][1]
                old_r = current[i][0]
                # Try a different residue
                new_r = random.randint(0, m_cur - 1)
                if new_r == old_r:
                    continue
                # Check validity
                ok = True
                for j, (r2, m2) in enumerate(current):
                    if j == i:
                        continue
                    g = gcd_cache[m_cur][m2]
                    if (new_r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    current[i][0] = new_r
                    moved = True
            
        if not moved:
            # Stuck? Restart with a new random clique
            new_clique = _random_valid_clique()
            if len(new_clique) > best_len:
                best_len = len(new_clique)
                best_clique = [c[:] for c in new_clique]
            current = [c[:] for c in new_clique]
            current_used = set(m for r, m in current)

    return best_clique

def construct():
    t0 = time.time()
    
    # Generate a few initial random cliques
    cliques = []
    for _ in range(10):
        if time.time() > t0 + 5.0: # Spend max 5s on initialization
            break
        c = _random_valid_clique()
        cliques.append(c)
        
    # Also add the deterministic "all even" one as a strong seed
    even_moduli = list(range(2, MAX_MOD + 1, 2))
    # Greedy assign for evens
    res_even = []
    for m in even_moduli:
        for r in range(m):
            ok = True
            for r2, m2 in res_even:
                g = gcd(m, m2)
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                res_even.append([r, m])
                break
    if res_even:
        cliques.append(res_even)
        
    if not cliques:
        return [[0, 2]]
        
    best = _local_search(cliques, t0, DEADLINE)
    
    # Final validation
    if _is_valid(best):
        return best
    else:
        # Fallback to the best valid one from initialization if search corrupted it (shouldn't happen)
        for c in cliques:
            if _is_valid(c):
                return c
        return [[0, 2]]
# EVOLVE-BLOCK-END
