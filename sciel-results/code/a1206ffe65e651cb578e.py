import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_valid_residue(classes, m):
        """Find a valid residue for modulus m given existing classes.
        Returns the residue that maximizes future flexibility by minimizing
        the number of future moduli that would be blocked."""
        mi = mod_idx[m]
        valid_residues = []
        for r in range(m):
            if can_add(classes, m, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return None
        
        # Score each valid residue by how many future small moduli it blocks
        # A residue r for modulus m blocks modulus m2 if for all r2 in [0,m2),
        # (r - r2) % gcd(m,m2) != 0 has no solution, i.e., r % gcd(m,m2) == r2 % gcd(m,m2)
        # Actually, it blocks m2 if the congruence class r mod gcd(m,m2) covers all
        # residues mod m2 that are compatible... Let me think differently.
        # 
        # For a given (m, r), a future modulus m2 can use residue r2 iff
        # (r - r2) % gcd(m, m2) != 0
        # So m2 is blocked if for ALL r2 in [0, m2), (r - r2) % gcd(m,m2) == 0
        # This happens iff gcd(m, m2) == m2 (i.e., m2 | m) AND r % m2 == r2 for the only
        # possible r2... Actually no.
        #
        # m2 is completely blocked if the set {r2 : (r-r2) % g == 0} where g = gcd(m,m2)
        # covers all of [0, m2). This set has size m2/g. It covers all of [0,m2) iff
        # m2/g == m2, i.e., g == 1. Wait, that's wrong.
        #
        # The forbidden residues for m2 are those r2 where (r - r2) % g == 0,
        # i.e., r2 ≡ r (mod g). There are m2/g such residues in [0, m2).
        # m2 is blocked iff m2/g == m2, i.e., g == 1. But if g == 1, then there's
        # only 1 forbidden residue, so m2 is NOT blocked.
        #
        # So no single class blocks another entirely. Instead, we should measure
        # how many valid residues remain for important future moduli.
        
        # Heuristic: prefer residues that leave more options for small moduli
        # For each valid residue, count how many small moduli (20-50) still have
        # at least one valid residue.
        best_r = valid_residues[0]
        best_flex = -1
        
        # Only check a subset of future moduli for speed
        future_moduli = [m2 for m2 in moduli if m2 > m][:30]
        
        for r in valid_residues[:50]:  # Limit to first 50 valid residues for speed
            flex = 0
            for m2 in future_moduli:
                g = gcd_table[mi][mod_idx[m2]]
                # Count valid residues for m2 given (m, r)
                # r2 is valid if (r - r2) % g != 0
                # Number of invalid r2: those where r2 ≡ r (mod g), count = m2 // g
                invalid_count = m2 // g
                valid_count = m2 - invalid_count
                if valid_count > 0:
                    flex += 1
            if flex > best_flex:
                best_flex = flex
                best_r = r
        
        return best_r
    
    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    def greedy_construct(order=None):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted(moduli, key=lambda m: 1/m)  # smallest moduli first
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r = find_valid_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes
    
    def local_search(classes, max_iter=300):
        """Try to improve by adding classes or performing swaps."""
        best_score = score(classes)
        best_classes = [list(c) for c in classes]
        used_moduli = set(m for m, r in best_classes)
        
        iter_count = 0
        while iter_count < max_iter:
            if time.time() > deadline:
                break
            iter_count += 1
            
            improved = False
            
            # Try adding a new class (smallest moduli first for better score)
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                r = find_valid_residue(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = score(new_classes)
                    if new_score > best_score:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break
            
            # Try 1-for-2 swap: remove one class, try to add two better ones
            if not improved and len(best_classes) > 2:
                # Sort current classes by weight ascending (remove low-weight first)
                sorted_by_weight = sorted(best_classes, key=lambda c: c[0], reverse=True)
                
                for m_remove in sorted_by_weight[:5]:  # Try removing up to 5 largest moduli
                    if time.time() > deadline:
                        break
                    temp_classes = [c for c in best_classes if c[0] != m_remove]
                    temp_used = set(m for m, r in temp_classes)
                    
                    added_any = False
                    # Try to add the smallest available moduli
                    for m1 in sorted(moduli):
                        if m1 in temp_used or m1 == m_remove:
                            continue
                        r1 = find_valid_residue(temp_classes, m1)
                        if r1 is not None:
                            temp_classes2 = temp_classes + [[m1, r1]]
                            temp_used2 = temp_used | {m1}
                            
                            for m2 in sorted(moduli):
                                if m2 in temp_used2 or m2 == m_remove:
                                    continue
                                r2 = find_valid_residue(temp_classes2, m2)
                                if r2 is not None:
                                    new_classes = temp_classes2 + [[m2, r2]]
                                    new_score = score(new_classes)
                                    if new_score > best_score:
                                        best_classes = new_classes
                                        best_score = new_score
                                        used_moduli = set(m for m, r in best_classes)
                                        improved = True
                                        added_any = True
                                        break
                        if added_any:
                            break
                    if added_any:
                        break
            
            if not improved:
                break
        
        return best_classes
    
    # Multi-start with different orderings
    best_classes = None
    best_score = Fraction(0, 1)
    
    # Start with greedy (smallest moduli first)
    classes = greedy_construct()
    classes = local_search(classes, max_iter=200)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Try randomized greedy multiple times
    random.seed(42)
    num_starts = 50
    for start in range(num_starts):
        if time.time() > deadline:
            break
        
        # Create a random ordering with strong bias toward small moduli
        # Use exponential weighting
        weights = [1.0 / (m ** 2.0) for m in moduli]
        order = list(moduli)
        random.shuffle(order)
        # Sort by random priority weighted toward small moduli
        priorities = [random.random() * w for m, w in zip(moduli, weights)]
        order = [m for _, m in sorted(zip(priorities, moduli), key=lambda x: x[0])]
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=100)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final polish: try to add any remaining small moduli
    if best_classes is not None:
        used = set(m for m, r in best_classes)
        for m in sorted(moduli):
            if m in used:
                continue
            r = find_valid_residue(best_classes, m)
            if r is not None:
                best_classes.append([m, r])
                used.add(m)
    
    return best_classes
