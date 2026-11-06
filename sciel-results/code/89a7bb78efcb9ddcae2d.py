"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations
import random

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Key insight: If gcd(m1, m2) == 1, the classes can NEVER be disjoint.
    So we only consider moduli that share factors with others in the set.
    
    Strategy:
    1. Precompute gcd relationships between all moduli in [20, 200].
    2. Multiple construction strategies with local search:
       a. Greedy by ascending m (smaller = higher weight)
       b. Group by common prime factor, pick best multiples
       c. Local search with add, swap, and 2-opt moves
    3. Return best found within time budget.
    """
    deadline = time.time() + 148.0  # Safety margin before 150s limit
    
    all_moduli = list(range(20, 201))
    n_moduli = len(all_moduli)
    
    # Precompute gcd table
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for j, m2 in enumerate(all_moduli):
            if i != j:
                gcd_table[(m1, m2)] = gcd(m1, m2)
    
    def get_gcd(m1, m2):
        return gcd_table.get((m1, m2), 1)
    
    def find_valid_residue(m, current_classes):
        """Find a residue r in [0, m) such that (m, r) is disjoint from all current classes."""
        for r in range(m):
            ok = True
            for cm, cr in current_classes:
                g = get_gcd(m, cm)
                if g == 1:
                    return None  # Can never be disjoint
                if r % g == cr % g:
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def find_best_residue(m, current_classes):
        """Find the residue r that maximizes flexibility for future additions."""
        best_r = None
        best_flex = -1
        for r in range(m):
            ok = True
            flex = 0
            for cm, cr in current_classes:
                g = get_gcd(m, cm)
                if g == 1:
                    return None
                if r % g == cr % g:
                    ok = False
                    break
                # Flexibility: how many other moduli could still be added
                flex += 1
            if ok and flex > best_flex:
                best_flex = flex
                best_r = r
        return best_r
    
    def calc_score(cls):
        """Calculate sum of 1/m for the given classes."""
        total = Fraction(0, 1)
        for m, r in cls:
            total += Fraction(1, m)
        return total
    
    def calc_score_float(cls):
        """Calculate sum of 1/m as float for quick comparison."""
        return sum(1.0 / m for m, r in cls)
    
    def is_pairwise_valid(cls):
        """Check if all pairs are disjoint."""
        n = len(cls)
        for i in range(n):
            for j in range(i+1, n):
                m1, r1 = cls[i]
                m2, r2 = cls[j]
                g = get_gcd(m1, m2)
                if g == 1:
                    return False
                if r1 % g == r2 % g:
                    return False
        return True
    
    def local_search(initial_classes, time_limit):
        """Improve a set of classes using local search moves."""
        current_classes = [c[:] for c in initial_classes]
        current_score = calc_score(current_classes)
        current_score_f = calc_score_float(current_classes)
        best_score = current_score
        best_score_f = current_score_f
        best_classes = [c[:] for c in current_classes]
        
        if not is_pairwise_valid(current_classes):
            return best_classes, best_score
        
        in_set = set(c[0] for c in current_classes)
        no_improve_count = 0
        
        while time.time() < time_limit:
            improved = False
            no_improve_count += 1
            
            # Move 1: Try adding a missing modulus
            for m in sorted(all_moduli, key=lambda x: -1.0/x):
                if m in in_set:
                    continue
                r = find_valid_residue(m, current_classes)
                if r is not None:
                    new_score_f = current_score_f + 1.0/m
                    if new_score_f > current_score_f + 1e-12:
                        current_classes.append([m, r])
                        current_score = current_score + Fraction(1, m)
                        current_score_f = new_score_f
                        in_set.add(m)
                        improved = True
                        if current_score_f > best_score_f:
                            best_score_f = current_score_f
                            best_score = current_score
                            best_classes = [c[:] for c in current_classes]
            
            # Move 2: Try swapping - replace a low-weight modulus with a higher-weight one
            if not improved:
                for i in range(len(current_classes)):
                    m_old = current_classes[i][0]
                    r_old = current_classes[i][1]
                    # Try all possible replacements, sorted by weight descending
                    candidates = [m for m in all_moduli if m not in in_set and m != m_old and 1.0/m > 1.0/m_old]
                    candidates.sort(key=lambda x: -1.0/x)
                    
                    found_swap = False
                    for m_new in candidates:
                        candidate = [c[:] for c in current_classes]
                        candidate.pop(i)
                        r_new = find_valid_residue(m_new, candidate)
                        if r_new is not None:
                            new_score_f = current_score_f - 1.0/m_old + 1.0/m_new
                            if new_score_f > current_score_f + 1e-12:
                                current_classes = candidate
                                current_classes.append([m_new, r_new])
                                current_score = current_score - Fraction(1, m_old) + Fraction(1, m_new)
                                current_score_f = new_score_f
                                in_set.remove(m_old)
                                in_set.add(m_new)
                                improved = True
                                found_swap = True
                                if current_score_f > best_score_f:
                                    best_score_f = current_score_f
                                    best_score = current_score
                                    best_classes = [c[:] for c in current_classes]
                                break
                    
                    if found_swap:
                        break
            
            # Move 3: 2-opt - remove two moduli, add two different ones
            if not improved and len(current_classes) >= 2:
                found_2opt = False
                # Try removing pairs of low-weight moduli
                indices = list(range(len(current_classes)))
                indices.sort(key=lambda i: 1.0/current_classes[i][0])  # Lowest weight first
                
                for i in range(min(len(indices), 5)):
                    for j in range(i+1, min(len(indices), 5)):
                        m1_old = current_classes[indices[i]][0]
                        m2_old = current_classes[indices[j]][0]
                        
                        candidate = [c[:] for c in current_classes]
                        # Remove the two moduli (higher index first)
                        if indices[i] > indices[j]:
                            candidate.pop(indices[i])
                            candidate.pop(indices[j])
                        else:
                            candidate.pop(indices[j])
                            candidate.pop(indices[i])
                        
                        cand_in_set = set(c[0] for c in candidate)
                        
                        # Try adding two new moduli
                        new_candidates = [m for m in all_moduli if m not in cand_in_set and m != m1_old and m != m2_old]
                        new_candidates.sort(key=lambda x: -1.0/x)
                        
                        for m_a in new_candidates[:10]:
                            r_a = find_valid_residue(m_a, candidate)
                            if r_a is None:
                                continue
                            cand2 = candidate + [[m_a, r_a]]
                            cand2_in_set = cand_in_set | {m_a}
                            
                            for m_b in new_candidates[:10]:
                                if m_b == m_a or m_b in cand2_in_set:
                                    continue
                                r_b = find_valid_residue(m_b, cand2)
                                if r_b is None:
                                    continue
                                
                                new_score_f = current_score_f - 1.0/m1_old - 1.0/m2_old + 1.0/m_a + 1.0/m_b
                                if new_score_f > current_score_f + 1e-12:
                                    current_classes = cand2
                                    current_classes.append([m_b, r_b])
                                    current_score = current_score - Fraction(1, m1_old) - Fraction(1, m2_old) + Fraction(1, m_a) + Fraction(1, m_b)
                                    current_score_f = new_score_f
                                    in_set = set(c[0] for c in current_classes)
                                    improved = True
                                    found_2opt = True
                                    if current_score_f > best_score_f:
                                        best_score_f = current_score_f
                                        best_score = current_score
                                        best_classes = [c[:] for c in current_classes]
                                    break
                            if found_2opt:
                                break
                        
                        if found_2opt:
                            break
                    
                    if found_2opt:
                        break
            
            # Move 4: Residue optimization - try to find better residues for existing moduli
            if not improved:
                for i in range(len(current_classes)):
                    m_i = current_classes[i][0]
                    r_old = current_classes[i][1]
                    
                    # Try all residues for this modulus
                    for r_new in range(m_i):
                        if r_new == r_old:
                            continue
                        # Check if new residue is valid with all others
                        valid = True
                        for j in range(len(current_classes)):
                            if i == j:
                                continue
                            m_j = current_classes[j][0]
                            r_j = current_classes[j][1]
                            g = get_gcd(m_i, m_j)
                            if g == 1:
                                valid = False
                                break
                            if r_new % g == r_j % g:
                                valid = False
                                break
                        
                        if valid:
                            # This residue is valid; we prefer it if it allows more future additions
                            # For now, just accept any valid different residue as it doesn't change score
                            # But it might help with future moves
                            current_classes[i][1] = r_new
                            improved = True
                            break
            
            if not improved:
                # Try randomized restarts of local search
                if no_improve_count > 3 and time.time() < time_limit - 5:
                    # Perturb: randomly change a residue or swap a modulus
                    if len(current_classes) > 0:
                        idx = random.randint(0, len(current_classes)-1)
                        m = current_classes[idx][0]
                        # Try a random valid residue
                        for _ in range(50):
                            r_rand = random.randint(0, m-1)
                            valid = True
                            for j in range(len(current_classes)):
                                if j == idx:
                                    continue
                                g = get_gcd(m, current_classes[j][0])
                                if g == 1:
                                    valid = False
                                    break
                                if r_rand % g == current_classes[j][1] % g:
                                    valid = False
                                    break
                            if valid:
                                current_classes[idx][1] = r_rand
                                no_improve_count = 0
                                break
                    else:
                        break
                else:
                    break
        
        return best_classes, best_score
    
    def greedy_ascending():
        """Greedy: add moduli in ascending order (smallest first = highest weight)."""
        classes = []
        in_set = set()
        for m in all_moduli:  # Already ascending
            r = find_valid_residue(m, classes)
            if r is not None:
                classes.append([m, r])
                in_set.add(m)
        return classes
    
    def greedy_by_prime_groups():
        """Group moduli by their smallest prime factor, pick best from each group."""
        from math import isqrt
        
        def smallest_prime_factor(n):
            if n % 2 == 0:
                return 2
            for i in range(3, isqrt(n) + 1, 2):
                if n % i == 0:
                    return i
            return n
        
        # Group moduli by smallest prime factor
        groups = {}
        for m in all_moduli:
            spf = smallest_prime_factor(m)
            if spf not in groups:
                groups[spf] = []
            groups[spf].append(m)
        
        # Try different strategies for combining groups
        best_classes = []
        best_score = Fraction(0, 1)
        
        for strategy in range(3):
            classes = []
            in_set = set()
            
            if strategy == 0:
                # Pick smallest from each group
                for spf in sorted(groups.keys()):
                    for m in sorted(groups[spf]):
                        if m in in_set:
                            continue
                        r = find_valid_residue(m, classes)
                        if r is not None:
                            classes.append([m, r])
                            in_set.add(m)
                            break
            elif strategy == 1:
                # Pick up to 2 from each group
                for spf in sorted(groups.keys()):
                    count = 0
                    for m in sorted(groups[spf]):
                        if m in in_set:
                            continue
                        r = find_valid_residue(m, classes)
                        if r is not None:
                            classes.append([m, r])
                            in_set.add(m)
                            count += 1
                            if count >= 2:
                                break
            else:
                # Greedy overall, but prefer moduli with more shared factors
                def factor_count(m):
                    cnt = 0
                    for other in all_moduli:
                        if other != m and get_gcd(m, other) > 1:
                            cnt += 1
                    return cnt
                
                for m in sorted(all_moduli, key=lambda x: (-factor_count(x), x)):
                    if m in in_set:
                        continue
                    r = find_valid_residue(m, classes)
                    if r is not None:
                        classes.append([m, r])
                        in_set.add(m)
            
            if classes:
                score = calc_score(classes)
                if score > best_score:
                    best_score = score
                    best_classes = [c[:] for c in classes]
        
        return best_classes
    
    # Run multiple strategies
    best_overall = []
    best_overall_score = Fraction(0, 1)
    
    strategies = [
        ("greedy_ascending", greedy_ascending),
        ("greedy_by_prime_groups", greedy_by_prime_groups),
    ]
    
    # Add a randomized greedy strategy
    random.seed(42)
    
    def greedy_randomized():
        """Greedy with randomized order, multiple restarts."""
        best = []
        best_score = Fraction(0, 1)
        
        for restart in range(10):
            classes = []
            in_set = set()
            order = all_moduli[:]
            random.shuffle(order)
            # But bias toward smaller moduli
            order.sort(key=lambda m: (random.random() * 0.3 + 1.0/m, m))
            
            for m in order:
                if m in in_set:
                    continue
                r = find_valid_residue(m, classes)
                if r is not None:
                    classes.append([m, r])
                    in_set.add(m)
            
            if classes:
                score = calc_score(classes)
                if score > best_score:
                    best_score = score
                    best = [c[:] for c in classes]
        
        return best
    
    strategies.append(("greedy_randomized", greedy_randomized))
    
    for name, func in strategies:
        if time.time() > deadline - 60:
            break
        try:
            classes = func()
            if classes and is_pairwise_valid(classes):
                score = calc_score(classes)
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall = [c[:] for c in classes]
        except Exception:
            pass
    
    # Local search on the best initial solution
    if best_overall and time.time() < deadline - 10:
        improved_classes, improved_score = local_search(
            best_overall, 
            time.time() + (deadline - time.time() - 5)
        )
        if improved_score > best_overall_score:
            best_overall = improved_classes
            best_overall_score = improved_score
    
    # Final validation
    if best_overall:
        if not is_pairwise_valid(best_overall):
            # Fallback to simple valid solution
            best_overall = greedy_ascending()
            if not is_pairwise_valid(best_overall):
                best_overall = []
    
    return [c[:] for c in best_overall]

# EVOLVE-BLOCK-END
