from math import gcd
import time
import random
from fractions import Fraction

def _is_valid(classes):
    """Check pairwise disjointness."""
    n = len(classes)
    for i in range(n):
        m1, r1 = classes[i]
        for j in range(i + 1, n):
            m2, r2 = classes[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                return False
    return True

def _score(classes):
    """Sum of 1/m as float."""
    return sum(1.0 / m for m, r in classes)

def _greedy_construct(order, deadline):
    """
    Greedily add moduli in given order. For each modulus, find a residue
    that is disjoint with all previously selected.
    """
    selected = []
    for m in order:
        if time.time() > deadline:
            break
        # Find a residue r in [0, m) such that for all (m_i, r_i) in selected:
        # r % gcd(m, m_i) != r_i % gcd(m, m_i)
        found = False
        for r in range(m):
            valid = True
            for (m_i, r_i) in selected:
                g = gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                selected.append([m, r])
                found = True
                break
        # If no residue found, skip this modulus
    return selected

def _local_search(classes, deadline):
    """
    Local search: try to add unselected moduli by adjusting residues of
    existing ones, and try swaps to free up capacity.
    """
    classes = [list(c) for c in classes]
    selected_mods = set(m for m, r in classes)
    all_mods = list(range(20, 201))
    unselected = [m for m in all_mods if m not in selected_mods]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each unselected modulus
        for m in unselected[:]:
            if time.time() > deadline:
                break
            # Try to find a residue for m that works with current selection
            for r in range(m):
                valid = True
                for (m_i, r_i) in classes:
                    g = gcd(m, m_i)
                    if r % g == r_i % g:
                        valid = False
                        break
                if valid:
                    classes.append([m, r])
                    unselected.remove(m)
                    selected_mods.add(m)
                    improved = True
                    break
        
        # Try swapping residues of existing classes to free capacity
        if not improved:
            # Try removing one class and re-adding two
            for idx in range(len(classes)):
                if time.time() > deadline:
                    break
                m_remove = classes[idx][0]
                saved = classes.pop(idx)
                # Try to add the removed one back with different residue, or add others
                # Try adding unselected moduli
                added_any = False
                for m in unselected[:]:
                    for r in range(m):
                        valid = True
                        for (m_i, r_i) in classes:
                            g = gcd(m, m_i)
                            if r % g == r_i % g:
                                valid = False
                                break
                        if valid:
                            classes.append([m, r])
                            unselected.remove(m)
                            selected_mods.add(m)
                            added_any = True
                            break
                    if added_any:
                        break
                # Also try re-adding the removed one
                if not added_any:
                    for r in range(m_remove):
                        valid = True
                        for (m_i, r_i) in classes:
                            g = gcd(m_remove, m_i)
                            if r % g == r_i % g:
                                valid = False
                                break
                        if valid:
                            classes.append([m_remove, r])
                            added_any = True
                            break
                if not added_any:
                    # Restore
                    classes.insert(idx, saved)
    
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_classes = None
    best_score = 0.0
    
    # Strategy 1: Greedy by descending 1/m (i.e., ascending m)
    order1 = list(range(20, 201))
    classes1 = _greedy_construct(order1, deadline)
    classes1 = _local_search(classes1, deadline)
    if _is_valid(classes1):
        s1 = _score(classes1)
        if s1 > best_score:
            best_score = s1
            best_classes = classes1
    
    # Strategy 2: Use pairwise coprime moduli (primes and prime powers)
    # This makes gcd=1 for all pairs, so any residues work
    # Include small moduli first for high weight
    coprime_mods = []
    used_primes = set()
    for m in range(20, 201):
        # Check if m is coprime to all previously selected
        is_coprime = True
        for cm in coprime_mods:
            if gcd(m, cm) != 1:
                is_coprime = False
                break
        if is_coprime:
            coprime_mods.append(m)
    
    # For coprime moduli, any residue works, so just pick r=0
    classes2 = [[m, 0] for m in coprime_mods]
    if _is_valid(classes2):
        s2 = _score(classes2)
        if s2 > best_score:
            best_score = s2
            best_classes = classes2
    
    # Strategy 3: Greedy with random residue ordering + local search
    random.seed(42)
    for attempt in range(5):
        if time.time() > deadline:
            break
        order3 = list(range(20, 201))
        random.shuffle(order3)
        # Sort by 1/m descending (smaller m first) but with some randomness
        order3.sort(key=lambda m: (random.random() * 0.1, m))
        classes3 = _greedy_construct(order3, deadline)
        classes3 = _local_search(classes3, deadline)
        if _is_valid(classes3):
            s3 = _score(classes3)
            if s3 > best_score:
                best_score = s3
                best_classes = classes3
    
    # Strategy 4: Start with primes, then greedily add composites
    primes_in_range = [m for m in range(20, 201) if all(m % p != 0 for p in range(2, int(m**0.5)+1))]
    classes4 = [[m, 0] for m in primes_in_range]
    # Now greedily add composites
    for m in range(20, 201):
        if m in primes_in_range:
            continue
        if time.time() > deadline:
            break
        for r in range(m):
            valid = True
            for (m_i, r_i) in classes4:
                g = gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                classes4.append([m, r])
                break
    classes4 = _local_search(classes4, deadline)
    if _is_valid(classes4):
        s4 = _score(classes4)
        if s4 > best_score:
            best_score = s4
            best_classes = classes4
    
    if best_classes is None:
        # Fallback: seed solution
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
    
    return best_classes
