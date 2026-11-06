"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    n = 192
    k = 3
    
    def greedy_select(order):
        """Greedy selection following the given order with efficient checking."""
        selected = []
        sel_set = set()
        
        for x in order:
            if x in sel_set:
                continue
            
            valid = True
            
            # Check 1: For each existing a in selected, does a | (p+q+x)
            # for any pair (p,q) from selected excluding a?
            for a in selected:
                others = [v for v in selected if v != a]
                found = False
                for i in range(len(others)):
                    for j in range(i+1, len(others)):
                        if (others[i] + others[j] + x) % a == 0:
                            found = True
                            break
                    if found:
                        break
                if found:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: Does x | (a+b+c) for any triple from selected?
            # For each pair (a,b), check if there's c in selected with c ≡ -(a+b) (mod x)
            found = False
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    a_val, b_val = selected[i], selected[j]
                    remainder = (a_val + b_val) % x
                    target = (x - remainder) % x
                    # Check if any c in selected (c != a, c != b) has c % x == target
                    for c_val in selected:
                        if c_val != a_val and c_val != b_val and c_val % x == target:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if found:
                continue
            
            selected.append(x)
            sel_set.add(x)
        
        return selected
    
    # Try different orderings
    best = []
    
    # Descending
    result = greedy_select(range(n, 0, -1))
    best = max(best, result, key=len)
    
    # Ascending
    result = greedy_select(range(1, n+1))
    best = max(best, result, key=len)
    
    # Random orders with fixed seed
    random.seed(42)
    for _ in range(200):
        order = list(range(1, n+1))
        random.shuffle(order)
        result = greedy_select(order)
        best = max(best, result, key=len)
    
    return best

# EVOLVE-BLOCK-END
