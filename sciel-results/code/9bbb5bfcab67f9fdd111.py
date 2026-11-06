"""Rewritten program with optimized incremental validity check."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    def can_add(selected_list, candidate):
        """Check if candidate can be added to selected_list."""
        s = list(selected_list)
        n = len(s)
        
        # Check: does candidate divide sum of any 3 from s?
        for combo in combinations(s, 3):
            if sum(combo) % candidate == 0:
                return False
        
        # Check: for each a in s, does a divide sum of any 3 from (s\{a}) ∪ {candidate}?
        for i, a in enumerate(s):
            others = [s[j] for j in range(n) if j != i]
            # 3-sums from others ∪ {candidate}: 
            # Case 1: all 3 from others (already checked if s was valid)
            # Case 2: 2 from others + candidate
            # Case 3: 1 from others + 2*... no, distinct
            # Actually need: 3 distinct from (others + [candidate])
            # Subsets of size 3 from others ∪ {candidate}:
            # - 3 from others: already valid since s was valid
            # - 2 from others + candidate
            # - 1 from others + 2 from... no, candidate appears once
            # So only need: 2 from others + candidate
            
            for j in range(len(others)):
                for k in range(j+1, len(others)):
                    total = others[j] + others[k] + candidate
                    if total % a == 0:
                        return False
        
        return True
    
    # Greedy from high to low
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if can_add(selected, x):
            selected.append(x)
    
    if len(selected) >= 48:
        return selected[:48]
    
    # Try different orderings if needed
    # Try from 192 down but skip some
    selected2 = []
    for x in range(192, 1, -1):
        if time.time() > deadline:
            break
        if len(selected2) >= 48:
            break
        if can_add(selected2, x):
            selected2.append(x)
    
    if len(selected2) > len(selected):
        selected = selected2
    
    if len(selected) >= 48:
        return selected[:48]
    
    # Local search: try to swap elements
    best = list(selected)
    best_len = len(best)
    
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            if best_len >= 48:
                break
            
            improved = False
            for i, a in enumerate(best):
                if time.time() > deadline:
                    break
                # Try removing a and adding b, c
                without_a = [v for v in best if v != a]
                for b in range(192, 0, -1):
                    if b in without_a:
                        continue
                    if can_add(without_a, b):
                        with_b = without_a + [b]
                        for c in range(192, 0, -1):
                            if c in with_b:
                                continue
                            if can_add(with_b, c):
                                best = with_b + [c]
                                best_len = len(best)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    if len(best) >= 48:
        return best[:48]
    
    return best

# EVOLVE-BLOCK-END
