import math
import time
import sys
from functools import lru_cache

def construct():
    deadline = time.time() + 115  # 150s limit, use 115s to be safe
    
    # Precompute gcd table
    max_m = 200
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(20, max_m + 1):
        for j in range(20, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Moduli in order of decreasing weight (1/m), i.e., increasing m
    # We process in order of increasing m (decreasing 1/m) for the bound
    moduli = list(range(20, 201))
    weights = {m: 1.0 / m for m in moduli}
    
    # Precompute: for each pair (m1, m2), the set of residue pairs that conflict
    # r1 % gcd(m1,m2) == r2 % gcd(m1,m2)
    
    # For the upper bound: given a set of selected (m, r) pairs,
    # a modulus m is "blocked" if for ALL r in [0, m), there exists some
    # selected (m2, r2) such that r % gcd(m, m2) == r2 % gcd(m, m2).
    # If not blocked, it can contribute 1/m to the bound.
    
    # Branch and bound
    best_score = 0.0
    best_selection = []
    
    # Order: process moduli in decreasing 1/m (increasing m) for the search tree
    # Actually, for B&B, we want to decide on high-weight items first.
    # So process in order of decreasing 1/m = increasing m? No, decreasing 1/m means larger m.
    # Wait: 1/m is largest for smallest m. So we want to decide small m first.
    # Process in order of increasing m.
    
    search_order = list(range(20, 201))  # increasing m
    
    def is_valid(m, r, selected):
        """Check if residue r for modulus m is valid given selected (m2, r2) pairs."""
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True
    
    def compute_upper_bound(idx, selected):
        """
        Upper bound on additional score from moduli at indices idx..end.
        For each modulus m at index >= idx, check if there exists any valid residue.
        If yes, add 1/m to the bound.
        """
        bound = 0.0
        for i in range(idx, len(search_order)):
            m = search_order[i]
            # Check if any residue r in [0, m) is valid
            found = False
            for r in range(m):
                if is_valid(m, r, selected):
                    found = True
                    break
            if found:
                bound += weights[m]
        return bound
    
    # To speed up: precompute for each modulus m, the list of residues
    # And use a smarter conflict check
    
    # Better approach: represent selected as a list, and for the bound,
    # use a faster check. Let's precompute for each m, which residues are available
    # given a set of selected classes.
    
    # Actually, let's use a different representation for speed.
    # For each modulus m, maintain the set of forbidden residues.
    # A residue r is forbidden for m if there exists selected (m2, r2) with r % gcd(m,m2) == r2 % gcd(m,m2).
    
    # Let's try a recursive B&B with memoization on the "state" of forbidden residues.
    # But the state space is huge. Let's just do B&B with good pruning.
    
    # To make it faster, let's limit the search depth or use a time-based cutoff.
    
    def branch_and_bound(idx, selected, current_score):
        nonlocal best_score, best_selection
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        # Update best if current score is better
        if current_score > best_score:
            best_score = current_score
            best_selection = list(selected)
        
        if idx >= len(search_order):
            return
        
        # Upper bound pruning
        ub = current_score + compute_upper_bound(idx, selected)
        if ub <= best_score:
            return
        
        m = search_order[idx]
        
        # Option 1: skip this modulus
        branch_and_bound(idx + 1, selected, current_score)
        
        # Option 2: try each valid residue for this modulus
        # To find a good solution quickly, try residues in a smart order
        for r in range(m):
            if is_valid(m, r, selected):
                selected.append((m, r))
                branch_and_bound(idx + 1, selected, current_score + weights[m])
                selected.pop()
        
    # Start B&B
    branch_and_bound(0, [], 0.0)
    
    # Convert best_selection to the required format [[m, r], ...]
    result = [[m, r] for m, r in best_selection]
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            m1, r1 = result[i]
            m2, r2 = result[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                # Invalid, should not happen
                return []
    
    return result
