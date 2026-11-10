"""Combined program for the no-divide-sum-of-three-others task."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 192
    
    # Precompute for each value a (1..n), the set of residues (b+c) % a
    # for all pairs b,c from 1..n, b != c, b != a, c != a
    # Actually we need: for a given set S and element a in S, check if there
    # exist b,c in S\{a} such that (b+c) % a == target.
    # We can precompute for each a, a dictionary mapping residue -> list of pairs?
    # Or better: for each a, precompute a bitset of achievable residues from pairs in 1..n excluding a.
    # But S is a subset, so we need to check against S specifically.
    
    # Alternative: precompute for each a, for each possible b (1..n, b!=a), 
    # what residues (b+c)%a are achievable as c varies over 1..n, c!=a, c!=b.
    # Then for a given S, we check: for each b in S\{a}, does (target - b) % a 
    # appear in the achievable residues for that b?
    
    # Let's precompute: residue_achieved[a][b] = set of residues (b+c)%a for c in 1..n, c!=a, c!=b
    # This is O(n^3) which is 192^3 ≈ 7M, acceptable.
    
    residue_achieved = {}
    for a in range(1, n + 1):
        residue_achieved[a] = {}
        for b in range(1, n + 1):
            if b == a:
                continue
            residues = set()
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                residues.add((b + c) % a)
            residue_achieved[a][b] = residues
    
    # Also precompute: for each a, the set of all residues achievable by any pair (b,c) in 1..n excluding a
    # This is useful for a quick rejection: if target not in any achievable residue for a, skip.
    # But we still need to check against specific S.
    
    def can_add(S, x):
        """
        Checks if x can be added to set S without violating the condition.
        Condition: No element in S U {x} divides the sum of any 3 distinct other elements.
        """
        S_set = set(S)
        
        # 1. Check if x divides the sum of any 3 elements in S
        if len(S) >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. Check if any a in S divides the sum of x and any 2 other elements in S
        for a in S:
            if a == 1:
                # 1 divides everything. If |S\{1} U {x}| >= 3, invalid.
                # |S\{1}| = len(S) - 1 (since 1 is in S)
                # New others count = len(S) - 1 + 1 = len(S)
                # Need len(S) >= 3 for 3 others.
                if len(S) >= 3:
                    return False
                continue
            
            target = (-x) % a
            
            # Need to check if there exist b, c in S\{a}, b != c, such that (b+c) % a == target
            # Use precomputed residue_achieved[a][b]
            S_minus_a = [v for v in S if v != a]
            if len(S_minus_a) < 2:
                continue
            
            for b in S_minus_a:
                if target in residue_achieved[a][b]:
                    # There exists some c in 1..n (c!=a, c!=b) such that (b+c)%a == target
                    # But we need c to be in S\{a}\{b}
                    # So we need to verify that at least one such c is in S
                    # Quick check: is there a c in S_minus_a, c != b, such that (b+c)%a == target?
                    found = False
                    for c in S_minus_a:
                        if c == b:
                            continue
                        if (b + c) % a == target:
                            found = True
                            break
                    if found:
                        return False
        
        return True

    best_set = []
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    # 1. Greedy Descending
    S1 = []
    unselected = set(range(1, n + 1))
    for x in range(n, 0, -1):
        if x in unselected:
            if can_add(S1, x):
                S1.append(x)
                unselected.remove(x)
    update_best(S1)

    # 2. Greedy Ascending
    S2 = []
    unselected = set(range(1, n + 1))
    for x in range(1, n + 1):
        if x in unselected:
            if can_add(S2, x):
                S2.append(x)
                unselected.remove(x)
    update_best(S2)

    # 3. Random Greedy
    for _ in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S3 = []
        unselected = set(order)
        for x in order:
            if x in unselected:
                if can_add(S3, x):
                    S3.append(x)
                    unselected.remove(x)
        update_best(S3)

    # 4. Local Search on Best Set
    if best_set:
        current_set = best_set[:]
        unselected = set(range(1, n + 1)) - set(current_set)
        
        attempts = 0
        while time.time() < deadline and attempts < 50000:
            attempts += 1
            if not current_set or not unselected:
                break
            
            # Random Swap
            idx = random.randint(0, len(current_set) - 1)
            a = current_set[idx]
            b = random.choice(list(unselected))
            
            # Check if swap is valid
            S_prime = [x for x in current_set if x != a]
            
            if can_add(S_prime, b):
                current_set[idx] = b
                unselected.remove(b)
                unselected.add(a)
                
                # Try to add more after swap
                for cand in list(unselected):
                    if can_add(current_set, cand):
                        current_set.append(cand)
                        unselected.remove(cand)
                        update_best(current_set)
                        break
                
    return best_set
# EVOLVE-BLOCK-END
