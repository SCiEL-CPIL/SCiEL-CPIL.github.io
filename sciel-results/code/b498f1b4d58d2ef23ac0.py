"""Model-and-optimize approach for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Moduli range
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
    
    # Build conflict graph
    # Nodes: (m_idx, r) where m_idx is index in moduli, r is residue
    # Edge between (i, r1) and (j, r2) if r1 ≡ r2 (mod gcd(moduli[i], moduli[j]))
    
    # For each pair of moduli, determine which residue pairs conflict
    # conflict[i][j][r1][r2] = True if residues conflict
    
    # Actually, let's build the graph more efficiently
    # For each modulus i, and each residue r1, find all (j, r2) that conflict
    
    # Let's use a different approach: branch and bound on the MWIS
    
    # First, let's create a list of all nodes with their weights
    # Node: (m_idx, r), weight = 1/moduli[m_idx]
    
    # For branch and bound, we need:
    # 1. A way to quickly check if adding a node conflicts with selected nodes
    # 2. An upper bound on the remaining weight
    
    # Let's precompute for each node, which other nodes it conflicts with
    
    # Total nodes: sum of moduli = sum(20..200) = 200*221/2 - 19*20/2 = 22100 - 190 = 21910
    # That's a lot. Let's think of a smarter approach.
    
    # Actually, the key insight is that for a given set of selected moduli,
    # we can determine valid residues via constraint propagation.
    
    # Let's use a hybrid approach:
    # 1. Start with a good initial solution from greedy
    # 2. Use local search with better neighborhood moves
    
    def can_add(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_best_residue(classes, m):
        """Find the best residue for modulus m given existing classes."""
        mi = mod_idx[m]
        valid_residues = []
        for r in range(m):
            if can_add(classes, m, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return None, 0
        
        # Score each valid residue by slack
        best_r = valid_residues[0]
        best_slack = -1
        
        for r in valid_residues:
            min_dist = m
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    min_dist = min(min_dist, dist)
            if min_dist > best_slack:
                best_slack = min_dist
                best_r = r
        
        return best_r, best_slack
    
    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        total = 0.0
        for (m, r) in classes:
            total += 1.0 / m
        return total
    
    def greedy_construct(order=None):
        """Greedy construction: add classes in the given order if possible."""
        if order is None:
            order = sorted(moduli)  # smallest moduli first
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r, slack = find_best_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        return classes
    
    def try_add_all(classes):
        """Try to add all remaining moduli, starting with smallest."""
        used_moduli = set(m for m, r in classes)
        remaining = [m for m in sorted(moduli) if m not in used_moduli]
        
        for m in remaining:
            if time.time() - start_time > deadline - 2:
                break
            r, slack = find_best_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
        
        return classes
    
    def local_search(classes, max_iter=1000):
        """Improve by trying to add missing moduli, swap residues, or swap moduli."""
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            
            improved = False
            
            # 1. Try adding a new modulus
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                r, slack = find_best_residue(best_classes, m)
                if r is not None:
                    new_classes = best_classes + [[m, r]]
                    new_score = best_score + 1.0/m
                    if new_score > best_score + 1e-12:
                        best_classes = new_classes
                        best_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break
            
            # 2. Try Swap-Add: Remove m_old, Add m_new if 1/m_new > 1/m_old
            if not improved:
                missing = [m for m in sorted(moduli) if m not in used_moduli]
                
                for m_new in missing:
                    if time.time() - start_time > deadline - 3:
                        break
                    
                    found_swap = False
                    for i in range(len(best_classes)):
                        m_old, r_old = best_classes[i]
                        if m_old <= m_new:
                            continue
                        
                        temp_classes = [list(c) for c in best_classes if c[0] != m_old]
                        r_new, _ = find_best_residue(temp_classes, m_new)
                        if r_new is not None:
                            new_classes = temp_classes + [[m_new, r_new]]
                            new_score = score(new_classes)
                            if new_score > best_score + 1e-12:
                                best_classes = new_classes
                                best_score = new_score
                                used_moduli = set(m for m, r in best_classes)
                                improved = True
                                found_swap = True
                                break
                    
                    if found_swap:
                        break
            
            # 3. Try residue optimization: for each class, try all residues to improve slack
            if not improved:
                for i in range(len(best_classes)):
                    m, r = best_classes[i]
                    other_classes = [c for j, c in enumerate(best_classes) if j != i]
                    new_r, _ = find_best_residue(other_classes, m)
                    if new_r is not None and new_r != r:
                        # Check if this improves overall slack (heuristic: just accept if valid)
                        # This is a neutral move in terms of score, but might help future additions
                        # For now, skip this to save time
                        pass
            
            if not improved:
                break
        
        return best_classes
    
    # Multi-start approach
    best_classes = None
    best_score = 0.0
    
    # Start 1: Greedy with smallest moduli first
    classes = greedy_construct()
    classes = try_add_all(classes)
    classes = local_search(classes, max_iter=1000)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Additional starts with different orderings
    import random
    random.seed(42)
    num_starts = 50
    
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline - 20:
            break
        
        # Create a random ordering biased toward small moduli
        remaining = list(moduli)
        order = []
        while remaining:
            alpha = 1.0 + start_idx * 0.1
            weights = [1.0 / (m ** alpha) for m in remaining]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx])
            remaining.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = try_add_all(classes)
        
        if time.time() - start_time > deadline - 10:
            s = score(classes)
            if s > best_score:
                best_score = s
                best_classes = [list(c) for c in classes]
            continue
        
        classes = local_search(classes, max_iter=500)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final polish
    if best_classes is not None:
        for round_num in range(5):
            if time.time() - start_time > deadline - 2:
                break
            max_iter = 500 + round_num * 300
            improved_classes = local_search(best_classes, max_iter=max_iter)
            improved_score = score(improved_classes)
            if improved_score > best_score + 1e-12:
                best_classes = improved_classes
                best_score = improved_score
            else:
                break
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
