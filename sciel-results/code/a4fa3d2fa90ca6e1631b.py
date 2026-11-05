import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g
    
    # For each pair (i,j), determine which residues r_i, r_j are incompatible
    # Two residue classes (r_i mod m_i) and (r_j mod m_j) are disjoint iff
    # (r_i - r_j) % gcd(m_i, m_j) != 0
    # So they conflict iff (r_i - r_j) % g == 0 where g = gcd(m_i, m_j)
    
    # Precompute for each pair (i,j) the set of residue pairs that conflict
    # Actually, for a given pair (i,j) with gcd g, the conflict condition is:
    # r_i ≡ r_j (mod g)
    # So for each pair, we can precompute which residues of i conflict with which residues of j
    
    # For efficiency, let's think about this differently.
    # For a fixed selection of moduli with assigned residues, a new modulus k with residue r
    # is compatible iff for all selected (i, r_i): (r - r_i) % gcd(m_k, m_i) != 0
    
    # Let's use a smarter approach: model as a graph problem
    # Each "node" is a (modulus_idx, residue) pair
    # Two nodes are adjacent if they are compatible (disjoint classes)
    # We want the maximum independent set... wait, no.
    # We want to select one node per chosen modulus such that all selected nodes are pairwise compatible.
    # This is a maximum clique in the compatibility graph, where we can only pick one node per modulus.
    # Actually, it's more like: we have groups (one per modulus), and within each group we can pick at most one.
    # We want the largest set of nodes, one per group, all pairwise compatible.
    
    # Let's use a backtracking approach with MRV (Minimum Remaining Values) heuristic
    # and constraint propagation.
    
    # First, let's try a greedy approach with multiple orderings, then local search.
    
    def check_residue(idx, r, selected):
        """Check if residue r for modulus idx is compatible with all selected (idx2, r2)"""
        for (idx2, r2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True
    
    # Greedy placement with a given order of moduli
    def try_greedy(order):
        selected = []
        for idx in order:
            m = moduli[idx]
            # Try residues in a smart order
            # Residue 0 often causes most conflicts, so try others first
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if check_residue(idx, r, selected):
                    selected.append((idx, r))
                    break
        return selected
    
    best = []
    best_len = 0
    
    # Deterministic orderings
    orderings = [
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        # Prefer moduli with many small prime factors first (harder to place)
        sorted(range(n), key=lambda i: (-math.gcd(moduli[i], 2**7), moduli[i])),
        # Prefer prime moduli first (easier to place)
        sorted(range(n), key=lambda i: (1 if all(moduli[i] % p != 0 for p in [2,3,5,7,11,13,17,19,23]) else 0, moduli[i])),
    ]
    
    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)
    
    # Random restarts with greedy
    random.seed(42)
    n_restarts = 2000
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)
    
    # Local search: try to add more moduli
    if time.time() < deadline - 15:
        selected_idx = set(idx for (idx, _) in best)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for idx in range(n):
                if idx in selected_idx:
                    continue
                m = moduli[idx]
                for r in range(m):
                    if check_residue(idx, r, best):
                        best.append((idx, r))
                        selected_idx.add(idx)
                        improved = True
                        break
                if improved:
                    break
    
    # Local search: try swapping residues to enable adding new moduli
    if time.time() < deadline - 5:
        selected_idx = set(idx for (idx, _) in best)
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                # Try changing one existing residue to free up space for new_idx
                for pos in range(len(best)):
                    old_idx, old_r = best[pos]
                    old_m = moduli[old_idx]
                    # Try all residues for old_idx
                    for new_r in range(old_m):
                        if new_r == old_r:
                            continue
                        # Check if new_r is compatible with all other selected (excluding old_idx)
                        ok_old = True
                        for qpos in range(len(best)):
                            if qpos == pos:
                                continue
                            other_idx, other_r = best[qpos]
                            g = gcd_tab[old_idx][other_idx]
                            if (new_r - other_r) % g == 0:
                                ok_old = False
                                break
                        if not ok_old:
                            continue
                        # Now check if new_idx can be added with some residue
                        found_new = False
                        for r_new in range(m_new):
                            ok_new = True
                            for qpos in range(len(best)):
                                if qpos == pos:
                                    if (r_new - new_r) % gcd_tab[new_idx][old_idx] == 0:
                                        ok_new = False
                                        break
                                else:
                                    other_idx, other_r = best[qpos]
                                    if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                        ok_new = False
                                        break
                            if ok_new:
                                found_new = True
                                break
                        if found_new:
                            best[pos] = (old_idx, new_r)
                            best.append((new_idx, r_new))
                            selected_idx.add(new_idx)
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break
    
    return [[r, moduli[idx]] for (idx, r) in best]
