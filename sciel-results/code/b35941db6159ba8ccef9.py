"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction

def construct():
    """
    Solve the Maximum Weight Independent Set problem on the conflict graph of (m, r) pairs
    using Branch and Bound.
    
    Nodes: (m, r) for m in [20, 200], r in [0, m).
    Weight: 1/m.
    Conflict: (m1, r1) and (m2, r2) conflict if r1 == r2 mod gcd(m1, m2).
    
    Strategy:
    1. Sort moduli by weight (1/m) descending.
    2. Backtracking search:
       - At each step i, consider modulus m_i.
       - Option A: Exclude m_i.
       - Option B: Include m_i. If included, find all valid residues r for m_i 
         that are disjoint from all previously selected (m_j, r_j).
         For each valid r, recurse.
    3. Upper Bound: Current Score + Sum of weights of remaining moduli.
    """
    
    # 1. Generate and sort moduli
    moduli = list(range(20, 201))
    # Sort by 1/m descending, which is equivalent to m ascending
    # But wait, 1/20 > 1/21 > ... so we want smallest m first.
    # The list is already ascending.
    
    N = len(moduli)
    weights = [Fraction(1, m) for m in moduli]
    
    # Precompute suffix sums of weights for upper bound
    # suffix_sum[i] = sum(weights[i:])
    suffix_sum = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
    best_score = Fraction(0)
    best_solution = []
    
    # To speed up residue checking, we can maintain the current list of (m, r)
    
    def valid_residues(m_new, current_classes):
        """
        Returns a list of valid residues r in [0, m_new) such that
        (m_new, r) is disjoint from all (m, r) in current_classes.
        """
        # We can start with all residues and remove invalid ones.
        # Or, check each residue.
        # m_new is at most 200. Checking 200 residues against <= 181 classes is fast.
        valid = []
        for r in range(m_new):
            is_ok = True
            for m_old, r_old in current_classes:
                g = math.gcd(m_new, m_old)
                if (r % g) == (r_old % g):
                    is_ok = False
                    break
            if is_ok:
                valid.append(r)
        return valid

    def dfs(i, current_score, current_classes):
        nonlocal best_score, best_solution
        
        # Pruning: Upper bound
        if current_score + suffix_sum[i] <= best_score:
            return
            
        if i == N:
            if current_score > best_score:
                best_score = current_score
                best_solution = [c[:] for c in current_classes]
            return
        
        m_i = moduli[i]
        w_i = weights[i]
        
        # Option 1: Include m_i
        # Find valid residues
        # Optimization: If current_classes is empty, all residues are valid.
        # But we only need to branch on *one* valid residue? 
        # No, different residues might allow different future moduli.
        # However, due to symmetry? 
        # If we pick r=0 vs r=1, do they lead to different feasibility for future moduli?
        # Yes, because gcd(m_future, m_i) might be > 1, and r mod g matters.
        # So we must try all valid residues.
        
        valid_rs = valid_residues(m_i, current_classes)
        
        for r in valid_rs:
            current_classes.append([m_i, r])
            dfs(i + 1, current_score + w_i, current_classes)
            current_classes.pop()
            
        # Option 2: Exclude m_i
        dfs(i + 1, current_score, current_classes)
        
    dfs(0, Fraction(0), [])
    
    return best_solution

# EVOLVE-BLOCK-END
