import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # gcd[m1_idx][m2_idx]
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_disjoint_set(classes):
        """Check if all pairs are disjoint. classes: list of (m_idx, r)"""
        n = len(classes)
        for i in range(n):
            mi, ri = classes[i]
            for j in range(i + 1, n):
                mj, rj = classes[j]
                g = gcd_table[mi][mj]
                if ri % g == rj % g:
                    return False
        return True
    
    def can_add(classes, m_idx, r):
        """Check if adding (m_idx, r) to classes keeps them disjoint."""
        for mi, ri in classes:
            g = gcd_table[mi][m_idx]
            if ri % g == r % g:
                return False
        return True
    
    def score(classes):
        """Sum of 1/m for the classes."""
        total = Fraction(0, 1)
        for m_idx, r in classes:
            total += Fraction(1, moduli[m_idx])
        return total
    
    # Greedy construction: try to add moduli in increasing order (smaller m = larger 1/m)
    # For each modulus, try all residues and pick one that fits
    def greedy_construct():
        classes = []
        # Sort moduli by 1/m descending (i.e., m ascending)
        for m_idx in range(n_mod):
            m = moduli[m_idx]
            # Try residues 0..m-1, prefer smaller residues for consistency
            found = False
            for r in range(m):
                if can_add(classes, m_idx, r):
                    classes.append((m_idx, r))
                    found = True
                    break
            # If no residue works, skip this modulus
        return classes
    
    # Local search: try to swap or improve
    def local_search(classes, max_iter=100000):
        classes = list(classes)
        current_score = score(classes)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            # Try to add a missing modulus
            for m_idx in range(n_mod):
                if m_idx in [c[0] for c in classes]:
                    continue
                m = moduli[m_idx]
                for r in range(m):
                    if can_add(classes, m_idx, r):
                        new_classes = classes + [(m_idx, r)]
                        new_score = score(new_classes)
                        if new_score > current_score:
                            classes = new_classes
                            current_score = new_score
                            improved = True
                            break
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
            # Try to swap a class with a better one
            for i, (mi, ri) in enumerate(classes):
                m_old = moduli[mi]
                # Try replacing with a different modulus
                for m_idx in range(n_mod):
                    if m_idx == mi or m_idx in [c[0] for c in classes if c[0] != mi]:
                        continue
                    m_new = moduli[m_idx]
                    if m_new >= m_old:
                        continue  # Only consider smaller moduli (higher 1/m)
                    for r in range(m_new):
                        # Check if replacing classes[i] with (m_idx, r) keeps disjointness
                        test_classes = classes[:i] + [(m_idx, r)] + classes[i+1:]
                        if is_disjoint_set(test_classes):
                            new_score = score(test_classes)
                            if new_score > current_score:
                                classes = test_classes
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return classes, current_score
    
    # Build initial solution greedily
    best_classes = greedy_construct()
    best_score = score(best_classes)
    
    # Local search to improve
    if time.time() < deadline - 10.0:
        improved_classes, improved_score = local_search(best_classes)
        if improved_score > best_score:
            best_classes = improved_classes
            best_score = improved_score
    
    # Convert back to [[m, r], ...] format
    result = [[moduli[m_idx], r] for m_idx, r in best_classes]
    
    # Verify validity
    assert is_disjoint_set(best_classes), "Result is not pairwise disjoint!"
    assert len(set(moduli[m_idx] for m_idx, r in best_classes)) == len(best_classes), "Moduli not distinct!"
    for m_idx, r in best_classes:
        assert 20 <= moduli[m_idx] <= 200, f"Modulus {moduli[m_idx]} out of range"
        assert 0 <= r < moduli[m_idx], f"Residue {r} out of range for modulus {moduli[m_idx]}"
    
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of classes: {len(result)}")
    print(f"Score: {sum(1/m for m, r in result)}")
    print(f"Classes: {result[:10]}...")
