import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_moduli = len(moduli)
    
    # Precompute gcds
    gcd_matrix = {}
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            gcd_matrix[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
            
    def is_compatible(selected, m, r):
        for m_p, r_p in selected:
            g = gcd_matrix.get((m, m_p), math.gcd(m, m_p))
            if (r - r_p) % g == 0:
                return False
        return True

    def find_valid_residue_min_conflict(selected, m):
        """Find residue that minimizes future conflicts (maximizes flexibility)."""
        if not selected:
            return 0
        
        best_r = None
        best_score = -1
        
        for r in range(m):
            if not is_compatible(selected, m, r):
                continue
            
            # Score: count how many unselected moduli would still be compatible
            # with this choice. Higher is better (more flexibility).
            score = 0
            for m2 in moduli:
                if m2 == m:
                    continue
                if any(m_p == m2 for m_p, _ in selected):
                    continue
                g = gcd_matrix.get((m, m2), math.gcd(m, m2))
                # For each residue of m2, check if it would conflict with (m, r)
                # A residue r2 of m2 conflicts if (r2 - r) % g == 0
                # Number of conflicting residues = g
                # Number of non-conflicting residues = m2 - g
                score += (m2 - g)
            
            if score > best_score:
                best_score = score
                best_r = r
        
        return best_r

    def find_valid_residue_random(selected, m):
        if not selected:
            return 0
        valid_residues = []
        for r in range(m):
            if is_compatible(selected, m, r):
                valid_residues.append(r)
        if not valid_residues:
            return None
        return random.choice(valid_residues)

    best_solution = []
    
    # Identify primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    
    # Strategy: 
    # 1. Try to include all primes (they are always mutually disjoint).
    # 2. Add composites greedily with conflict-minimizing residue selection.
    
    # Let's run randomized greedy with different orders
    orders = []
    
    # 1. Primes first, then composites sorted by "difficulty" (high gcd with others first)
    for _ in range(200):
        comp = composites.copy()
        random.shuffle(comp)
        orders.append(primes + comp)
        
    # 2. Sort composites by number of divisors (fewer divisors = fewer conflicts?)
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
        
    div_count = {m: count_divisors(m) for m in moduli}
    
    # 3. Primes first, composites sorted by divisors (ascending)
    orders.append(primes + sorted(composites, key=lambda x: div_count[x]))
    orders.append(primes + sorted(composites, key=lambda x: -div_count[x]))
    
    # 4. Full random
    for _ in range(300):
        full = moduli.copy()
        random.shuffle(full)
        orders.append(full)
        
    # 5. Sort by max gcd with others (ascending) - "sparse" moduli first
    max_gcds = {m: 0 for m in moduli}
    for m in moduli:
        for m2 in moduli:
            if m != m2:
                g = gcd_matrix.get((m, m2), math.gcd(m, m2))
                if g > max_gcds[m]:
                    max_gcds[m] = g
    orders.append(sorted(moduli, key=lambda x: max_gcds[x]))
    orders.append(sorted(moduli, key=lambda x: -max_gcds[x]))
    
    # 6. Randomized sort by divisors
    for _ in range(200):
        shuffled_moduli = moduli.copy()
        random.shuffle(shuffled_moduli)
        orders.append(sorted(shuffled_moduli, key=lambda x: div_count[x] + random.uniform(-0.5, 0.5)))
        
    # 7. Randomized sort by max gcd
    for _ in range(200):
        shuffled_moduli = moduli.copy()
        random.shuffle(shuffled_moduli)
        orders.append(sorted(shuffled_moduli, key=lambda x: max_gcds[x] + random.uniform(-0.5, 0.5)))

    # Track which moduli have been tried with min-conflict strategy
    used_min_conflict = set()
    
    for order in orders:
        if time.time() > deadline:
            break
            
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            
            # Use min-conflict strategy for new moduli, random for others
            if m not in used_min_conflict:
                r = find_valid_residue_min_conflict(selected, m)
                if r is not None:
                    selected.append((m, r))
                    used_min_conflict.add(m)
            else:
                r = find_valid_residue_random(selected, m)
                if r is not None:
                    selected.append((m, r))
                
        if len(selected) > len(best_solution):
            best_solution = selected
            
    # Local Search: try to add any modulus not in the best solution
    if best_solution and time.time() < deadline - 5:
        selected_set = set(m for m, r in best_solution)
        for m in moduli:
            if time.time() > deadline - 2:
                break
            if m in selected_set:
                continue
            # Try to find a valid residue for m given current selection
            r = find_valid_residue_min_conflict(best_solution, m)
            if r is not None:
                best_solution.append((m, r))
                selected_set.add(m)
    
    result = [[r, m] for m, r in best_solution]
    
    if not result:
        return [[0, 2]]
        
    return result
