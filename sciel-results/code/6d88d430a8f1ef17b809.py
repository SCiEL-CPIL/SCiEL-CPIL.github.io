import time
import math
import random

def construct():
    deadline = time.time() + 140  # safety margin under 150s
    
    # Precompute gcd table for all pairs of even moduli in [20, 200]
    moduli = list(range(20, 201, 2))  # 100 even moduli
    n = len(moduli)
    idx = {m: i for i, m in enumerate(moduli)}
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    best_score = 0
    best_classes = []
    
    # Greedy: try adding moduli in order of 1/m (smallest first for max score)
    # We want to maximize sum of 1/m, so prioritize small moduli
    # But we need to find valid residues
    
    def try_greedy(order):
        """Try to build a solution using moduli in the given order."""
        classes = []
        for m in order:
            mi = idx.get(m)
            if mi is None:
                continue
            # Try all residues for this modulus
            for r in range(m):
                valid = True
                for m2, r2 in classes:
                    m2i = idx[m2]
                    g = gcd_table[mi][m2i]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    classes.append([m, r])
                    break
        return classes
    
    # Strategy 1: Greedy by smallest modulus first
    classes1 = try_greedy(sorted(moduli))
    score1 = sum(1.0 / m for m, _ in classes1)
    
    if score1 > best_score:
        best_score = score1
        best_classes = classes1
    
    # Strategy 2: Greedy but skip some moduli to allow better coverage
    # Try different starting points and skip patterns
    rng = random.Random(42)
    
    for attempt in range(3000):
        if time.time() > deadline - 5:
            break
        
        # Randomized greedy with some randomization in order
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        # Weight towards smaller moduli
        weighted = sorted(shuffled, key=lambda m: (m * 0.5 + rng.random() * 50))
        
        classes = try_greedy(weighted)
        score = sum(1.0 / m for m, _ in classes)
        
        if score > best_score:
            best_score = score
            best_classes = classes
    
    # Local search: try to improve by swapping residues or adding/removing moduli
    def can_add(classes, m_add, r_add):
        """Check if we can add modulus m_add with residue r_add."""
        m_add_i = idx.get(m_add)
        if m_add_i is None:
            return False
        for m2, r2 in classes:
            m2i = idx[m2]
            g = gcd_table[m_add_i][m2i]
            if r_add % g == r2 % g:
                return False
        return True
    
    def try_add_best(classes):
        """Try to add the best (smallest m) modulus that fits. Returns (new_classes, new_score, added)."""
        current_score = sum(1.0 / m for m, _ in classes)
        for m_add in moduli:
            if any(m_add == m2 for m2, _ in classes):
                continue
            m_add_i = idx.get(m_add)
            if m_add_i is None:
                continue
            for r_add in range(m_add):
                if can_add(classes, m_add, r_add):
                    new_classes = classes[:]
                    new_classes.append([m_add, r_add])
                    return new_classes, current_score + 1.0/m_add, True
        return classes, current_score, False
    
    def local_search(classes, max_iters=8000):
        """Improve via local search with random kick."""
        classes = [c[:] for c in classes]
        score = sum(1.0 / m for m, _ in classes)
        rng_local = random.Random(123)
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Move A: Try changing residue of an existing class to enable adding a new modulus
            for i in range(len(classes)):
                m, r = classes[i]
                mi = idx.get(m)
                if mi is None:
                    continue
                for new_r in range(m):
                    if new_r == r:
                        continue
                    valid = True
                    for j in range(len(classes)):
                        if j == i:
                            continue
                        m2, r2 = classes[j]
                        m2i = idx[m2]
                        g = gcd_table[mi][m2i]
                        if new_r % g == r2 % g:
                            valid = False
                            break
                    if valid:
                        old_r = classes[i][1]
                        classes[i] = [m, new_r]
                        new_classes, new_score, added = try_add_best(classes)
                        if added:
                            classes = new_classes
                            score = new_score
                            improved = True
                        else:
                            classes[i] = [m, old_r]
                        if improved:
                            break
                if improved:
                    break
            
            # Move B (Random Kick): If stuck, randomly perturb a residue then re-optimize
            if not improved:
                if len(classes) > 2:
                    i = rng_local.randint(0, len(classes)-1)
                    m, r = classes[i]
                    old_r = r
                    new_r = rng_local.randint(0, m-1)
                    classes[i] = [m, new_r]
                    # Try to add something
                    new_classes, new_score, added = try_add_best(classes)
                    if added:
                        classes = new_classes
                        score = new_score
                        improved = True
                    else:
                        # Try to restore original and see if we can add
                        classes[i] = [m, old_r]
                        new_classes, new_score, added2 = try_add_best(classes)
                        if added2:
                            classes = new_classes
                            score = new_score
                            improved = True
                        else:
                            # Keep the perturbation as a new starting point
                            classes[i] = [m, new_r]
            
            if not improved:
                break
        
        return classes, score
    
    improved_classes, improved_score = local_search(best_classes)
    if improved_score > best_score:
        best_score = improved_score
        best_classes = improved_classes
    
    # Second round of local search from the improved state
    improved_classes2, improved_score2 = local_search(best_classes)
    if improved_score2 > best_score:
        best_score = improved_score2
        best_classes = improved_classes2
    
    return best_classes
