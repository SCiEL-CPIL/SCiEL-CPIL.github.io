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
       c. Local search with add, swap, and remove-readd moves
    3. Return best found within time budget.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    all_moduli = list(range(20, 201))
    num_moduli = len(all_moduli)
    
    # Precompute gcd table for all pairs
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for j, m2 in enumerate(all_moduli):
            if j > i:
                gcd_table[(m1, m2)] = gcd(m1, m2)
                gcd_table[(m2, m1)] = gcd(m1, m2)
    
    # Precompute for each pair (m1, m2), the forbidden residues of r1 mod gcd(m1,m2)
    # that would make them collide. Actually, we need: for given (m, r) and existing (cm, cr),
    # r mod gcd(m, cm) != cr mod gcd(m, cm)
    
    def find_valid_residue_fast(m, current_classes):
        """
        Find a residue r in [0, m) such that (m, r) is disjoint from all current classes.
        Uses a marking array for speed.
        """
        if not current_classes:
            return 0
        
        # Check if m can coexist with any current class
        for cm, cr in current_classes:
            g = gcd(m, cm)
            if g == 1:
                return None
        
        # Mark forbidden residues
        # For each existing class (cm, cr), r must satisfy r % gcd(m, cm) != cr % gcd(m, cm)
        forbidden = [False] * m
        
        for cm, cr in current_classes:
            g = gcd(m, cm)
            if g == 1:
                return None
            forbidden_residue = cr % g
            # All r in [0, m) where r % g == forbidden_residue are forbidden
            for r in range(forbidden_residue, m, g):
                forbidden[r] = True
        
        # Find first valid residue
        for r in range(m):
            if not forbidden[r]:
                return r
        return None
    
    def calc_score(cls):
        """Calculate sum of 1/m for the given classes."""
        total = Fraction(0, 1)
        for m, r in cls:
            total += Fraction(1, m)
        return total
    
    def calc_score_float(cls):
        """Calculate sum of 1/m as float for quick comparisons."""
        return sum(1.0 / m for m, r in cls)
    
    def is_pairwise_valid(cls):
        """Check if all pairs are disjoint."""
        n = len(cls)
        for i in range(n):
            for j in range(i+1, n):
                m1, r1 = cls[i]
                m2, r2 = cls[j]
                g = gcd(m1, m2)
                if g == 1:
                    return False
                if r1 % g == r2 % g:
                    return False
        return True
    
    def local_search(initial_classes, time_limit):
        """Improve a set of classes using local search moves."""
        current_classes = [c[:] for c in initial_classes]
        current_score = calc_score(current_classes)
        best_score = current_score
        best_classes = [c[:] for c in current_classes]
        
        if not is_pairwise_valid(current_classes):
            return best_classes, best_score
        
        in_set = set(c[0] for c in current_classes)
        weights = {m: Fraction(1, m) for m in all_moduli}
        
        # Pre-sort moduli by weight descending for repeated use
        moduli_by_weight_desc = sorted(all_moduli, key=lambda x: -float(weights[x]))
        
        while time.time() < time_limit:
            improved = False
            
            # Move 1: Try adding a missing modulus
            for m in moduli_by_weight_desc:
                if m in in_set:
                    continue
                r = find_valid_residue_fast(m, current_classes)
                if r is not None:
                    new_classes = current_classes + [[m, r]]
                    new_score = current_score + weights[m]
                    if new_score > current_score:
                        current_classes = new_classes
                        current_score = new_score
                        in_set.add(m)
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_classes = [c[:] for c in current_classes]
                        break  # Restart after one addition
            
            # Move 2: Try 1-swap - replace one modulus with another
            if not improved and time.time() < time_limit:
                # Sort current classes by weight ascending (least valuable first)
                sorted_indices = sorted(range(len(current_classes)), 
                                       key=lambda i: float(weights[current_classes[i][0]]))
                
                for i in sorted_indices:
                    m_old = current_classes[i][0]
                    found_swap = False
                    
                    for m_new in moduli_by_weight_desc:
                        if m_new in in_set or m_new == m_old:
                            continue
                        if float(weights[m_new]) <= float(weights[m_old]):
                            break  # No point trying lower weight moduli
                    
                    candidate = [c[:] for c in current_classes]
                    candidate.pop(i)
                    r_new = find_valid_residue_fast(m_new, candidate)
                    if r_new is not None:
                        candidate.append([m_new, r_new])
                        new_score = current_score - weights[m_old] + weights[m_new]
                        if new_score > current_score:
                            current_classes = candidate
                            current_score = new_score
                            in_set.remove(m_old)
                            in_set.add(m_new)
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_classes = [c[:] for c in current_classes]
                            break  # Restart after one swap
                
                # Move 3: Try 2-swap - remove two, add two
                if not improved and time.time() < time_limit:
                    n = len(current_classes)
                    for i in range(n):
                        for j in range(i+1, n):
                            m_remove1 = current_classes[i][0]
                            m_remove2 = current_classes[j][0]
                            
                            candidate = [c[:] for c in current_classes]
                            candidate.pop(j)
                            candidate.pop(i)
                            
                            # Try adding two new moduli from top candidates
                            add_options = [m for m in moduli_by_weight_desc 
                                          if m not in set(c[0] for c in candidate)]
                            
                            found = False
                            for m_add1 in add_options[:15]:
                                r_add1 = find_valid_residue_fast(m_add1, candidate)
                                if r_add1 is None:
                                    continue
                                temp_candidate = candidate + [[m_add1, r_add1]]
                                
                                for m_add2 in add_options[:15]:
                                    if m_add2 == m_add1:
                                        continue
                                    r_add2 = find_valid_residue_fast(m_add2, temp_candidate)
                                    if r_add2 is None:
                                        continue
                                    
                                    new_candidate = temp_candidate + [[m_add2, r_add2]]
                                    new_score = (current_score 
                                                - weights[m_remove1] 
                                                - weights[m_remove2]
                                                + weights[m_add1] 
                                                + weights[m_add2])
                                    if new_score > current_score:
                                        current_classes = new_candidate
                                        current_score = new_score
                                        in_set = set(c[0] for c in current_classes)
                                        improved = True
                                        found = True
                                        if current_score > best_score:
                                            best_score = current_score
                                            best_classes = [c[:] for c in current_classes]
                                        break
                                if found:
                                    break
                        if improved:
                            break
            
            if not improved:
                break
        
        return best_classes, best_score
    
    def greedy_ascending():
        """Greedy: add moduli in ascending order (smallest first = highest weight)."""
        classes = []
        in_set = set()
        for m in all_moduli:  # Already ascending
            r = find_valid_residue_fast(m, classes)
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
        
        best_classes = []
        best_score = Fraction(0, 1)
        
        for strategy in range(4):
            classes = []
            in_set = set()
            
            if strategy == 0:
                # Pick smallest from each group
                for spf in sorted(groups.keys()):
                    for m in sorted(groups[spf]):
                        if m in in_set:
                            continue
                        r = find_valid_residue_fast(m, classes)
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
                        r = find_valid_residue_fast(m, classes)
                        if r is not None:
                            classes.append([m, r])
                            in_set.add(m)
                            count += 1
                            if count >= 2:
                                break
            elif strategy == 2:
                # Greedy overall, prefer moduli with more shared factors
                def factor_count(m):
                    cnt = 0
                    for other in all_moduli:
                        if other != m and gcd(m, other) > 1:
                            cnt += 1
                    return cnt
                
                for m in sorted(all_moduli, key=lambda x: (-factor_count(x), x)):
                    if m in in_set:
                        continue
                    r = find_valid_residue_fast(m, classes)
                    if r is not None:
                        classes.append([m, r])
                        in_set.add(m)
            else:
                # Greedy overall, prefer moduli that can coexist with many others
                def coexist_count(m, current):
                    cnt = 0
                    for cm in current:
                        if gcd(m, cm) > 1:
                            cnt += 1
                    return cnt
                
                for m in sorted(all_moduli):
                    if m in in_set:
                        continue
                    r = find_valid_residue_fast(m, classes)
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
            # Bias toward smaller moduli
            order.sort(key=lambda m: (random.random() * 0.2 + 1.0/m, m))
            
            for m in order:
                if m in in_set:
                    continue
                r = find_valid_residue_fast(m, classes)
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
        if time.time() > deadline - 30:
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
    
    # Final polish pass: try to add any remaining moduli
    if best_overall and time.time() < deadline - 2:
        in_set = set(c[0] for c in best_overall)
        for m in sorted(all_moduli, key=lambda x: -1.0/x):
            if m in in_set:
                continue
            r = find_valid_residue_fast(m, best_overall)
            if r is not None:
                best_overall.append([m, r])
                in_set.add(m)
    
    # Final validation
    if best_overall:
        if not is_pairwise_valid(best_overall):
            best_overall = greedy_ascending()
            if not is_pairwise_valid(best_overall):
                best_overall = []
    
    return [c[:] for c in best_overall]

# EVOLVE-BLOCK-END
