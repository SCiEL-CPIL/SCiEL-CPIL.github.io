"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # All possible moduli
    moduli = list(range(20, 201))
    n = len(moduli)
    
    # Precompute gcd matrix
    gcd_matrix = [[gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
    
    def is_compatible(selected, m_idx, r):
        """Check if residue r for moduli[m_idx] is compatible with all selected."""
        for (sel_idx, sel_r) in selected:
            g = gcd_matrix[m_idx][sel_idx]
            if r % g == sel_r % g:
                return False
        return True
    
    def score_of(selected):
        """Compute exact rational score."""
        s = Fraction(0, 1)
        for (idx, r) in selected:
            s += Fraction(1, moduli[idx])
        return s
    
    def greedy_construct(order):
        """Greedy: process moduli in given order, pick smallest valid residue."""
        selected = []
        selected_set = set()
        for m_idx in order:
            if time.time() > deadline:
                break
            m = moduli[m_idx]
            # Try residues 0, 1, ..., m-1
            for r in range(m):
                if is_compatible(selected, m_idx, r):
                    selected.append((m_idx, r))
                    selected_set.add(m_idx)
                    break
        return selected
    
    def local_search(selected, max_iters=5000):
        """Try to improve by removing and re-adding moduli, or swapping residues."""
        if not selected:
            return selected
        best_selected = list(selected)
        best_score = score_of(best_selected)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            current_score = score_of(best_selected)
            if current_score > best_score:
                best_score = current_score
            
            # Try to add a new modulus
            added = False
            for m_idx in range(n):
                if m_idx in [s[0] for s in best_selected]:
                    continue
                m = moduli[m_idx]
                for r in range(m):
                    if is_compatible(best_selected, m_idx, r):
                        new_selected = best_selected + [(m_idx, r)]
                        new_score = score_of(new_selected)
                        if new_score > best_score:
                            best_selected = new_selected
                            best_score = new_score
                            added = True
                            break
                if added:
                    break
            
            if not added:
                # Try to remove a low-value modulus and add a high-value one
                # Find the modulus with smallest weight (largest m)
                best_to_remove = max(range(len(best_selected)), key=lambda i: moduli[best_selected[i][0]])
                candidate = best_selected[best_to_remove]
                candidate_idx = candidate[0]
                candidate_weight = Fraction(1, moduli[candidate_idx])
                
                # Remove it
                temp_selected = [s for s in best_selected if s[0] != candidate_idx]
                temp_score = score_of(temp_selected)
                
                # Try to add a different modulus
                found_better = False
                for m_idx in range(n):
                    if m_idx == candidate_idx or m_idx in [s[0] for s in best_selected]:
                        continue
                    m = moduli[m_idx]
                    # Only consider if weight is better
                    if Fraction(1, m) <= candidate_weight:
                        continue
                    for r in range(m):
                        if is_compatible(temp_selected, m_idx, r):
                            new_selected = temp_selected + [(m_idx, r)]
                            new_score = score_of(new_selected)
                            if new_score > best_score:
                                best_selected = new_selected
                                best_score = new_score
                                found_better = True
                                break
                    if found_better:
                        break
                
                if not found_better:
                    # Try swapping residues of existing moduli
                    for i in range(len(best_selected)):
                        idx_i, r_i = best_selected[i]
                        for r_new in range(moduli[idx_i]):
                            if r_new == r_i:
                                continue
                            # Check if r_new is compatible with all other selected
                            other = [s for j, s in enumerate(best_selected) if j != i]
                            if is_compatible(other, idx_i, r_new):
                                new_selected = list(best_selected)
                                new_selected[i] = (idx_i, r_new)
                                new_score = score_of(new_selected)
                                if new_score > best_score:
                                    best_selected = new_selected
                                    best_score = new_score
                                    break
            else:
                continue
        
        return best_selected
    
    # Strategy 1: Greedy with natural order
    natural_order = list(range(n))
    result1 = greedy_construct(natural_order)
    result1 = local_search(result1, max_iters=2000)
    best_result = result1
    best_score = score_of(best_result)
    
    # Strategy 2: Greedy with random orders + local search
    rng = random.Random(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random ordering biased toward smaller moduli
        order = list(range(n))
        # Weight by inverse modulus for bias
        weights = [1.0 / moduli[i] for i in range(n)]
        # Sample without replacement using weights
        remaining = list(range(n))
        shuffled_order = []
        while remaining:
            total_w = sum(weights[i] for i in remaining)
            r = rng.random() * total_w
            cum = 0
            for idx, i in enumerate(remaining):
                cum += weights[i]
                if r <= cum:
                    shuffled_order.append(i)
                    remaining.pop(idx)
                    break
            else:
                shuffled_order.append(remaining.pop())
        
        result = greedy_construct(shuffled_order)
        result = local_search(result, max_iters=1000)
        s = score_of(result)
        if s > best_score:
            best_score = s
            best_result = result
    
    # Convert to required format
    result_list = [[moduli[idx], r] for (idx, r) in best_result]
    return result_list

# EVOLVE-BLOCK-END
