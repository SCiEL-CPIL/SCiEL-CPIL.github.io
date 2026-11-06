import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin (90s limit)
    
    n = 192
    max_size = 48
    
    # Precompute all possible triples of distinct elements from [1..192]
    # and their sums. We'll use this to quickly check divisibility.
    
    # For fast checking, precompute for each candidate a, 
    # which pairs (b,c) with b<c, b,c != a create a sum divisible by a
    # i.e., (b+c) % a == (a - candidate) % a... wait, let me think differently.
    
    # Strategy: Use bitmask representation for the selected set.
    # Precompute for each element a:
    #   bad_pairs[a] = set of (b,c) pairs such that (b+c+a) % a == 0, i.e., (b+c) % a == 0
    #   bad_triples[a] = set of triples (b,c,d) such that (b+c+d) % a == 0
    
    # Actually, let's use a different approach:
    # When checking if we can add candidate x to set S:
    # 1. x | (b+c+d) for any 3 distinct b,c,d in S  => (b+c+d) % x == 0
    # 2. For each a in S: a | (b+c+x) for any 2 distinct b,c in S, b,c != a => (b+c+x) % a == 0
    
    # Precompute: for each a in [1..192], for each sum s in [3..573], whether s % a == 0
    # But that's too much. Let's be smarter.
    
    # Precompute for each a: the set of residues r in [0, a-1] such that r == 0 mod a
    # i.e., we need (b+c) % a == (-x) % a = (a - x%a) % a
    
    # Let's precompute pair_sums_mod[a][r] = set of pairs (b,c) from [1..192], b<c, b,c!=a, 
    # such that (b+c) % a == r
    # This is too expensive to precompute fully.
    
    # Alternative: Use a greedy approach with efficient incremental checking.
    # Keep the set as a list, and for each new candidate, only check the new constraints.
    
    # For efficiency, let's precompute for each a:
    #   For each possible pair sum s = b+c where b,c in [1..192], b<c, b,c!=a:
    #     whether s % a == 0
    # But we can compute this on the fly since the set size is at most 48.
    
    def can_add(selected, candidate):
        """Check if candidate can be added to selected without violating constraints."""
        sel = list(selected)
        sel_set = set(sel)
        
        # Check 1: candidate divides sum of any 3 distinct elements in selected
        # (b+c+d) % candidate == 0 for distinct b,c,d in selected
        m = len(sel)
        if m >= 3:
            for i in range(m):
                for j in range(i+1, m):
                    for k in range(j+1, m):
                        if (sel[i] + sel[j] + sel[k]) % candidate == 0:
                            return False
        
        # Check 2: for each a in selected, a divides (b+c+candidate) for distinct b,c in selected, b,c != a
        for a in sel:
            # Find all pairs (b,c) from selected where b != a, c != a, b < c
            others = [x for x in sel if x != a]
            mo = len(others)
            if mo >= 2:
                for i in range(mo):
                    for j in range(i+1, mo):
                        if (others[i] + others[j] + candidate) % a == 0:
                            return False
        
        return True
    
    # Greedy: try to build a large set
    # Key insight: larger numbers are less likely to divide sums of smaller numbers
    # Also, numbers that are prime or have few divisors are better candidates
    
    best = []
    
    # Try descending order first (largest numbers first)
    for x in range(n, 0, -1):
        if time.time() > deadline - 3.0:
            break
        if can_add(best, x):
            best.append(x)
        if len(best) >= max_size:
            break
    
    current_best = len(best)
    best_set = list(best)
    
    # Try multiple random orderings
    candidates = list(range(1, n+1))
    
    if time.time() < deadline - 5.0:
        for restart in range(50):
            if time.time() > deadline - 3.0:
                break
            random.shuffle(candidates)
            curr = []
            for x in candidates:
                if len(curr) >= max_size:
                    break
                if can_add(curr, x):
                    curr.append(x)
            if len(curr) > current_best:
                current_best = len(curr)
                best_set = curr
    
    # Local search: try to swap elements to improve
    if time.time() < deadline - 5.0 and current_best < max_size:
        best_set = list(best_set)
        best_set_set = set(best_set)
        
        for iteration in range(200):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            # Try removing one element and adding another
            for i in range(len(best_set)):
                if time.time() > deadline - 3.0:
                    break
                removed = best_set[i]
                temp_set = best_set[:i] + best_set[i+1:]
                temp_set_set = set(temp_set)
                
                # Try adding a candidate not in the set
                for x in range(n, 0, -1):
                    if x in temp_set_set:
                        continue
                    if can_add(temp_set, x):
                        new_set = temp_set + [x]
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_set_set = set(best_set)
                            improved = True
                            break
                        elif len(new_set) == len(best_set):
                            # Same size, but maybe better for future additions
                            # Keep the one with larger minimum element (more room)
                            if min(new_set) > min(best_set):
                                best_set = new_set
                                best_set_set = set(best_set)
                                improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
    
    return best_set[:max_size]
