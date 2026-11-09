import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s kill

    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_moduli = len(moduli)

    # Precompute gcds
    gcd_matrix = {}
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            gcd_matrix[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    div_count = {m: count_divisors(m) for m in moduli}

    # Helper to check compatibility
    def is_compatible(selected, m, r):
        for m_p, r_p in selected:
            g = gcd_matrix.get((m, m_p), math.gcd(m, m_p))
            if (r - r_p) % g == 0:
                return False
        return True

    def find_valid_residue(selected, m, randomize=False):
        if not selected:
            return 0
        valid_residues = []
        for r in range(m):
            if is_compatible(selected, m, r):
                valid_residues.append(r)
        if not valid_residues:
            return None
        if randomize:
            return random.choice(valid_residues)
        return valid_residues[0]

    best_solution = []

    # Phase 1: Deterministic Greedy
    # Try various orderings
    orderings = [
        primes + composites,
        composites + primes,
        sorted(moduli, key=lambda x: div_count[x]),
        sorted(moduli, key=lambda x: -div_count[x]),
        sorted(moduli),
        sorted(moduli, reverse=True),
    ]

    for order in orderings:
        if time.time() > deadline - 10:
            break
        selected = []
        for m in order:
            r = find_valid_residue(selected, m, randomize=False)
            if r is not None:
                selected.append((m, r))
        if len(selected) > len(best_solution):
            best_solution = selected

    # Phase 2: Randomized Restart Greedy
    # Focus on "Primes First" as it seems robust
    while time.time() < deadline - 20:
        r_choice = random.random()
        if r_choice < 0.5:
            # Primes first, random composites
            comp = composites.copy()
            random.shuffle(comp)
            order = primes + comp
        elif r_choice < 0.8:
            # Random primes, then random composites
            pri = primes.copy()
            comp = composites.copy()
            random.shuffle(pri)
            random.shuffle(comp)
            order = pri + comp
        else:
            # Full random
            order = moduli.copy()
            random.shuffle(order)

        selected = []
        for m in order:
            # Check time less frequently
            if (len(selected) % 5 == 0) and time.time() > deadline - 20:
                break
            r = find_valid_residue(selected, m, randomize=True)
            if r is not None:
                selected.append((m, r))
        
        if len(selected) > len(best_solution):
            best_solution = selected

    # Phase 3: Local Search / Improvement
    # Try to add missing moduli or swap residues to allow more moduli
    # This is expensive, so limit it
    
    improved = True
    max_iter = 50
    iter_count = 0
    
    while improved and iter_count < max_iter and time.time() < deadline - 5:
        improved = False
        iter_count += 1
        
        selected_mods = set(m for m, r in best_solution)
        unselected_mods = [m for m in moduli if m not in selected_mods]
        
        # 1. Try to add any unselected modulus directly
        for m in unselected_mods:
            if time.time() > deadline - 5:
                break
            r = find_valid_residue(best_solution, m, randomize=True)
            if r is not None:
                best_solution.append((m, r))
                improved = True
                # Re-calculate unselected? Or just break and restart loop?
                # Breaking out of this inner loop to restart the while loop is safer
                break
        
        if improved:
            continue

        # 2. Try to swap a residue in best_solution to allow adding an unselected one
        # This is O(N^2 * M). Might be too slow if done naively for all pairs.
        # Let's try a limited random subset of swaps.
        
        # Pick a random unselected modulus to try to add
        if unselected_mods:
            target_m = random.choice(unselected_mods)
            
            # Try to find a swap that makes target_m addable
            # We can try changing the residue of a random selected modulus
            # and checking if target_m becomes compatible.
            
            # Optimization: Instead of trying all swaps, just try to find *any*
            # valid configuration for the current set + target_m via a small backtracking
            # or just random restarts focused on this specific addition?
            
            # Simpler: Try changing residues of existing moduli randomly to see if
            # it opens up space for target_m.
            
            for _ in range(20): # Try 20 random perturbations
                if time.time() > deadline - 5:
                    break
                
                # Pick a random index in best_solution
                idx = random.randint(0, len(best_solution) - 1)
                m_sel, r_sel = best_solution[idx]
                
                # Try a new random residue for m_sel
                new_r = random.randint(0, m_sel - 1)
                
                # Check if new_r is compatible with all OTHER selected moduli
                compatible = True
                for j, (m_p, r_p) in enumerate(best_solution):
                    if j == idx:
                        continue
                    g = gcd_matrix.get((m_sel, m_p), math.gcd(m_sel, m_p))
                    if (new_r - r_p) % g == 0:
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                # If compatible, swap it in
                old_pair = best_solution[idx]
                best_solution[idx] = (m_sel, new_r)
                
                # Now try to add target_m
                r_target = find_valid_residue(best_solution, target_m, randomize=True)
                if r_target is not None:
                    best_solution.append((target_m, r_target))
                    improved = True
                    break
                else:
                    # Revert swap if it didn't help
                    best_solution[idx] = old_pair

    result = [[r, m] for m, r in best_solution]
    
    if not result:
        return [[0, 2]]
        
    return result
