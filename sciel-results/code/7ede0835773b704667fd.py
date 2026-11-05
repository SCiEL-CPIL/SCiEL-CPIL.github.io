"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def construct():
    """
    Return a list of [residue, modulus] pairs with distinct moduli in [2,128],
    canonical residues 0<=r<m, such that every pair of infinite residue classes
    is disjoint (verified by exact gcd criterion). Maximize the number of classes.
    
    Two classes (r1, m1) and (r2, m2) are disjoint iff gcd(m1, m2) does not divide (r1 - r2).
    """
    start_time = time.time()
    deadline = 85  # 5s margin before 90s kill
    
    def check_disjoint(classes):
        """Check if all pairs are disjoint using exact gcd criterion."""
        n = len(classes)
        for i in range(n):
            r1, m1 = classes[i]
            for j in range(i + 1, n):
                r2, m2 = classes[j]
                g = math.gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True
    
    # Seed solution: powers of 2 with residue 2^k - 1
    def seed_solution():
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    best = seed_solution()
    best_count = len(best)
    
    # Precompute gcd for all pairs of moduli in [2, 128]
    all_moduli = list(range(2, 129))
    n_mods = len(all_moduli)
    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            g = math.gcd(all_moduli[i], all_moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Backtracking search with pruning
    # Order moduli by some heuristic
    # We'll try to build a set where every pair has gcd > 1 (necessary condition)
    
    # First, find all moduli that can potentially coexist (gcd > 1 with all others in the set)
    # This is a clique problem in the graph where edges connect moduli with gcd > 1
    
    # Build adjacency: can[i][j] = True if gcd(all_moduli[i], all_moduli[j]) > 1
    can = [[False] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            if gcd_mat[i][j] > 1:
                can[i][j] = True
                can[j][i] = True
    
    # Find maximum clique size as an upper bound
    # For now, just do a greedy clique finding
    def greedy_clique():
        remaining = set(range(n_mods))
        clique = []
        while remaining:
            # Pick the node with highest degree in remaining subgraph
            best_node = max(remaining, key=lambda x: sum(1 for y in remaining if can[x][y]))
            clique.append(best_node)
            remaining = {y for y in remaining if can[best_node][y]}
        return clique
    
    clique = greedy_clique()
    clique_moduli = [all_moduli[i] for i in clique]
    
    # Now do backtracking on this clique
    # Order by increasing number of compatible residues
    
    def num_compatible_residues(m_idx, current_classes):
        """Count how many residues for all_moduli[m_idx] are compatible with current_classes."""
        m = all_moduli[m_idx]
        count = 0
        for r in range(m):
            ok = True
            for r_j, m_j in current_classes:
                j_idx = all_moduli.index(m_j)
                g = gcd_mat[m_idx][j_idx]
                if (r - r_j) % g == 0:
                    ok = False
                    break
            if ok:
                count += 1
        return count
    
    # Backtracking with time limit
    def backtrack(current_classes, remaining_indices, depth=0):
        nonlocal best, best_count
        if time.time() > deadline:
            return
        
        if len(current_classes) > best_count:
            best = [list(c) for c in current_classes]
            best_count = len(best)
        
        if not remaining_indices:
            return
        
        # Try each remaining modulus
        for idx in list(remaining_indices):
            if time.time() > deadline:
                return
            
            m = all_moduli[idx]
            # Try to find a compatible residue
            found = False
            for r in range(m):
                ok = True
                for r_j, m_j in current_classes:
                    j_idx = all_moduli.index(m_j)
                    g = gcd_mat[idx][j_idx]
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    current_classes.append((r, m))
                    backtrack(current_classes, remaining_indices - {idx}, depth + 1)
                    current_classes.pop()
                    found = True
                    break  # Only try first valid residue for this modulus
            
            if not found:
                continue
    
    # Start backtracking from the clique, ordered by value
    clique_order = sorted(clique, key=lambda x: all_moduli[x])
    backtrack([], set(clique_order))
    
    return best
# EVOLVE-BLOCK-END
