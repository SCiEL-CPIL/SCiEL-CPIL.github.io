import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    n = 192
    max_size = 48
    
    # Precompute all possible triples sums? No, dynamic set changes.
    # We will maintain the set S and helper structures.
    
    def is_valid_set(S):
        """Check if S is a valid set."""
        S_list = list(S)
        k = len(S_list)
        if k < 4:
            return True
        
        # Check condition: a does not divide sum of any 3 distinct others
        # We can iterate over a in S, and check all triples from S \ {a}
        # This is O(K^4) which is 48^4 ~ 5M. Too slow for repeated checks in local search.
        
        # Efficient check:
        # 1. Precompute all sums of 3 distinct elements in S.
        # 2. For each a in S, check if a divides any sum in sums3 that does NOT include a?
        #    Wait, the condition is "sum of 3 DISTINCT OTHER".
        #    So the triple {b,c,d} must not contain a.
        #    If we precompute all sums of triples from S, we need to know which triples they come from.
        #    Or, simpler:
        #    For each a in S:
        #       Check all triples {b,c,d} from S\{a}.
        #       This is O(K * C(K-1, 3)).
        #       C(47, 3) = 16215. 48 * 16215 ~ 777k operations.
        #       This is acceptable for a few checks, but maybe not for thousands.
        
        pass

    def can_add(x, S, sums3, residues):
        """
        Check if x can be added to S.
        S: set of current elements
        sums3: set of all sums of 3 distinct elements in S
        residues: dict a -> set of (b+c) % a for b,c in S\{a}, b!=c
        """
        # Condition 1: x does not divide sum of any 3 distinct elements in S
        for s in sums3:
            if s % x == 0:
                return False
        
        # Condition 2: For each a in S, a does not divide (x + b + c) for any b,c in S\{a}
        # This is equivalent to: (b+c) % a != (-x) % a
        target_res = (-x) % max(S) if S else 0 # Modulo base doesn't matter, just need consistency per a
        
        for a in S:
            target = (-x) % a
            if target in residues[a]:
                return False
        
        return True

    def update_structures(S, x, added):
        """
        Update sums3 and residues when x is added to or removed from S.
        Returns new sums3 and residues.
        """
        # This is complex to do incrementally efficiently.
        # Given S size <= 48, recomputing is O(K^3) for sums3 and O(K^3) for residues.
        # 48^3 ~ 110k. Very fast.
        
        new_S = set(S)
        if added:
            new_S.add(x)
        else:
            new_S.discard(x)
        
        S_list = list(new_S)
        k = len(S_list)
        
        new_sums3 = set()
        if k >= 3:
            for t in combinations(S_list, 3):
                new_sums3.add(sum(t))
        
        new_residues = {}
        if k >= 3:
            for a in S_list:
                others = [v for v in S_list if v != a]
                res_set = set()
                # We need pairs (b,c) from others
                # Optimization: just iterate pairs
                for i in range(len(others)):
                    for j in range(i+1, len(others)):
                        res_set.add((others[i] + others[j]) % a)
                new_residues[a] = res_set
        
        return new_sums3, new_residues

    # Initial Greedy
    S = set()
    # Try descending order first
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        # Check if x can be added
        # Recompute structures for check? Or just brute force check for greedy?
        # Greedy is only done once or twice. Brute force is fine.
        
        # Brute force check for adding x to S
        valid = True
        S_list = list(S)
        # Check 1: x divides sum of 3 in S
        if len(S_list) >= 3:
            for t in combinations(S_list, 3):
                if sum(t) % x == 0:
                    valid = False
                    break
        if valid:
            # Check 2: a in S divides x + b + c
            for a in S_list:
                others = [v for v in S_list if v != a]
                if len(others) >= 2:
                    # Check if exists b,c in others s.t. (x+b+c) % a == 0
                    # b+c = k*a - x
                    # We can just check all pairs
                    for i in range(len(others)):
                        for j in range(i+1, len(others)):
                            if (x + others[i] + others[j]) % a == 0:
                                valid = False
                                break
                        if not valid:
                            break
                if not valid:
                    break
        
        if valid:
            S.add(x)
            
    if len(S) >= max_size:
        return list(S)
        
    # Local Search
    # Precompute structures for current S
    sums3, residues = update_structures(S, None, False)
    
    best_S = set(S)
    best_size = len(S)
    
    while time.time() < deadline:
        # Try to add
        added_any = False
        # Sort candidates by some heuristic? Random?
        candidates = [x for x in range(1, 193) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, S, sums3, residues):
                S.add(x)
                sums3, residues = update_structures(S, x, True)
                added_any = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                if len(S) >= max_size:
                    return list(S)
                break # Restart candidate loop with new structure
        
        if not added_any:
            # Try swaps
            # Pick a random y to remove
            if not S:
                break
            y = random.choice(list(S))
            # Tentatively remove y
            S.remove(y)
            sums3, residues = update_structures(S, y, False)
            
            # Try to add a random x
            candidates = [x for x in range(1, 193) if x not in S]
            random.shuffle(candidates)
            swapped = False
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(x, S, sums3, residues):
                    S.add(x)
                    sums3, residues = update_structures(S, x, True)
                    swapped = True
                    # Size is same, but set changed
                    # Check if this leads to better size? 
                    # We don't update best_S here as size didn't change, 
                    # but it might allow future additions.
                    break
            
            if not swapped:
                # Revert swap
                S.add(y)
                sums3, residues = update_structures(S, y, True)
            else:
                # Accept swap, continue loop to try adding more
                pass
                
    return list(best_S)
