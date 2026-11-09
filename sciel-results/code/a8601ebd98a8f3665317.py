"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _solve_residues(moduli, deadline):
    """
    Use backtracking to find valid residues for a given set of moduli.
    Returns a list of [residue, modulus] or None if no solution found within deadline.
    """
    n = len(moduli)
    if n == 0:
        return []
    
    # Precompute gcds between moduli
    gcds = [[1]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Order moduli by "tightness": smallest min-gcd first
    # If n is large, this is O(n^2)
    min_gcds = []
    for i in range(n):
        min_g = float('inf')
        for j in range(n):
            if i != j:
                if gcds[i][j] < min_g:
                    min_g = gcds[i][j]
        min_gcds.append(min_g)
        
    order = sorted(range(n), key=lambda i: min_gcds[i])
    
    # Map original indices to new order
    ordered_moduli = [moduli[i] for i in order]
    ordered_gcds = [[gcds[order[i]][order[j]] for j in range(n)] for i in range(n)]
    
    residues = [None] * n
    
    def backtrack(k):
        if time.time() > deadline:
            return False
        if k == n:
            return True
            
        m = ordered_moduli[k]
        
        # Heuristic: Try residues that are likely to be valid
        # We can check constraints against already assigned residues first.
        
        # To speed up, we can pre-check if any residue is possible.
        # But simple iteration is okay for small n.
        
        for r in range(m):
            ok = True
            for j in range(k):
                g = ordered_gcds[k][j]
                if (r - residues[j]) % g == 0:
                    ok = False
                    break
            if ok:
                residues[k] = r
                if backtrack(k+1):
                    return True
                residues[k] = None
        return False

    # Try a few random permutations of residue orders if deterministic fails?
    # For now, deterministic.
    if backtrack(0):
        # Map back to original order
        res_map = {}
        for new_idx, old_idx in enumerate(order):
            res_map[old_idx] = residues[new_idx]
            
        result = []
        for i in range(n):
            result.append([res_map[i], moduli[i]])
        return result
        
    return None

def _find_max_classes(max_mod=128, deadline=None):
    if deadline is None:
        deadline = time.time() + 85
    
    best_set = []
    best_size = 0
    
    all_moduli = list(range(2, max_mod+1))
    
    # Strategy: 
    # 1. Generate candidate cliques (sets of pairwise non-coprime moduli)
    # 2. Sort by size descending
    # 3. Try to solve CSP for each
    # 4. Return the largest solvable one
    
    def _greedy_clique(all_mods):
        # Order by degree in non-coprime graph
        degrees = {}
        for m in all_mods:
            deg = 0
            for m2 in all_mods:
                if m != m2 and gcd(m, m2) > 1:
                    deg += 1
            degrees[m] = deg
            
        sorted_mods = sorted(all_mods, key=lambda x: -degrees[x])
        
        clique = []
        for m in sorted_mods:
            if all(gcd(m, m2) > 1 for m2 in clique):
                clique.append(m)
        return clique

    # Generate several clique candidates
    candidates = []
    
    # Candidate 1: Greedy on all
    candidates.append(_greedy_clique(all_moduli))
    
    # Candidate 2: All evens
    evens = [m for m in all_moduli if m % 2 == 0]
    candidates.append(evens)
    
    # Candidate 3: All multiples of 3
    mult3 = [m for m in all_moduli if m % 3 == 0]
    candidates.append(mult3)
    
    # Candidate 4: All multiples of 6
    mult6 = [m for m in all_moduli if m % 6 == 0]
    candidates.append(mult6)
    
    # Candidate 5: Randomized Greedy (multiple times)
    for _ in range(100):
        random.shuffle(all_moduli)
        candidates.append(_greedy_clique(all_moduli))
        
    # Sort candidates by size descending
    candidates.sort(key=len, reverse=True)
    
    # Try to solve CSP for each candidate
    seen_cliques = set()
    
    for cand in candidates:
        if len(cand) <= best_size:
            continue
            
        # Check if it's a clique
        is_clique = True
        for i in range(len(cand)):
            for j in range(i+1, len(cand)):
                if gcd(cand[i], cand[j]) == 1:
                    is_clique = False
                    break
            if not is_clique:
                break
                
        if not is_clique:
            continue
            
        # Skip duplicates
        cand_tuple = tuple(sorted(cand))
        if cand_tuple in seen_cliques:
            continue
        seen_cliques.add(cand_tuple)
            
        # Try to solve
        sol = _solve_residues(cand, deadline)
        if sol is not None:
            if len(sol) > best_size:
                best_size = len(sol)
                best_set = sol
                
        if time.time() > deadline:
            break
            
    return best_set

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
