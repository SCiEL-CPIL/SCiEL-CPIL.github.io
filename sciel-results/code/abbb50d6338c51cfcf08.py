import time
import math

def construct():
    deadline = time.time() + 105  # safety margin under 120s
    
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
    import random
    rng = random.Random(42)
    
    for attempt in range(2000):
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
    def local_search(classes, max_iters=5000):
        """Improve via local search."""
        classes = [c[:] for c in classes]
        score = sum(1.0 / m for m, _ in classes)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            improved = False
            
            # Try changing residue of an existing class
            for i, (m, r) in enumerate(classes):
                mi = idx.get(m)
                if mi is None:
                    continue
                for new_r in range(m):
                    if new_r == r:
                        continue
                    valid = True
                    for j, (m2, r2) in enumerate(classes):
                        if j == i:
                            continue
                        m2i = idx[m2]
                        g = gcd_table[mi][m2i]
                        if new_r % g == r2 % g:
                            valid = False
                            break
                    if valid:
                        # Check if this is beneficial (it's neutral in score but may enable additions)
                        classes[i] = [m, new_r]
                        # Try to add a new modulus
                        for m_add in moduli:
                            if any(m_add == m2 for m2, _ in classes):
                                continue
                            m_add_i = idx.get(m_add)
                            if m_add_i is None:
                                continue
                            for r_add in range(m_add):
                                valid_add = True
                                for m2, r2 in classes:
                                    m2i = idx[m2]
                                    g = gcd_table[m_add_i][m2i]
                                    if r_add % g == r2 % g:
                                        valid_add = False
                                        break
                                if valid_add:
                                    new_classes = classes[:]
                                    new_classes.append([m_add, r_add])
                                    new_score = sum(1.0 / m for m, _ in new_classes)
                                    if new_score > score:
                                        classes = new_classes
                                        score = new_score
                                        improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return classes, score
    
    improved_classes, improved_score = local_search(best_classes)
    if improved_score > best_score:
        best_score = improved_score
        best_classes = improved_classes
    
    return best_classes
