"""Rewritten program: greedy + local-search improvement for pairwise disjoint residue classes."""
import time
from math import gcd

def _is_compatible(selected, m, r):
    """Check if (m, r) is disjoint from all entries in selected."""
    for sm, sr in selected:
        g = gcd(sm, m)
        if (r - sr) % g == 0:
            return False
    return True

def _score(selected):
    """Compute sum of 1/m for selected classes."""
    return sum(1.0 / m for m, r in selected)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Phase 1: Greedy selection by decreasing weight (increasing m is wrong; we want small m first for large 1/m)
    # Actually, to maximize sum of 1/m, we want to include as many small moduli as possible.
    # Process moduli in increasing order (20, 21, ..., 200), picking the smallest valid residue.
    selected = []
    
    for m in range(20, 201):
        if time.time() > deadline - 10.0:
            break
        found = False
        for r in range(m):
            if _is_compatible(selected, m, r):
                selected.append((m, r))
                found = True
                break
        # If no residue found, skip this modulus (rare given the structure)

    best_score = _score(selected)
    best_selected = list(selected)

    # Phase 2: Local search - try to swap residues or add/remove to improve
    # Try re-optimizing: for each selected class, try different residues to free up space
    # for additional classes, or to allow a better global configuration.
    
    # A more powerful approach: iterative improvement
    # Try removing one element and re-greedy-filling to see if we get a better set.
    # Also try swapping residues.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try each possible removal + re-greedy
        for i in range(len(best_selected)):
            if time.time() > deadline - 5.0:
                break
            
            # Remove element i
            candidate = [x for j, x in enumerate(best_selected) if j != i]
            
            # Re-greedy fill: try to add back all moduli not in candidate
            new_selected = list(candidate)
            existing_mods = set(m for m, r in new_selected)
            
            for m in range(20, 201):
                if m in existing_mods:
                    continue
                for r in range(m):
                    if _is_compatible(new_selected, m, r):
                        new_selected.append((m, r))
                        break
            
            new_score = _score(new_selected)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_selected = new_selected
                improved = True
                break
        
        # Also try residue swaps: for each pair, try changing one residue
        if not improved and time.time() < deadline - 5.0:
            for i in range(len(best_selected)):
                if time.time() > deadline - 5.0:
                    break
                m_i, r_i = best_selected[i]
                for new_r in range(m_i):
                    if new_r == r_i:
                        continue
                    # Check if new_r is compatible with all others
                    ok = True
                    for j in range(len(best_selected)):
                        if j == i:
                            continue
                        m_j, r_j = best_selected[j]
                        g = gcd(m_i, m_j)
                        if (new_r - r_j) % g == 0:
                            ok = False
                            break
                    if ok:
                        # This swap is valid; try it and see if it enables more additions
                        candidate = list(best_selected)
                        candidate[i] = (m_i, new_r)
                        existing_mods = set(m for m, r in candidate)
                        new_selected = list(candidate)
                        
                        for m in range(20, 201):
                            if m in existing_mods:
                                continue
                            for r in range(m):
                                if _is_compatible(new_selected, m, r):
                                    new_selected.append((m, r))
                                    break
                        
                        new_score = _score(new_selected)
                        if new_score > best_score + 1e-12:
                            best_score = new_score
                            best_selected = new_selected
                            improved = True
                            break
                if improved:
                    break

    # Convert to required format [[m, r], ...]
    return [[m, r] for m, r in best_selected]
