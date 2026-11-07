"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import random
from itertools import combinations

def construct():
    """Find largest subset of {1..128} with no three distinct elements a,b,c
    such that a+b, a+c, b+c are all in the set."""
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    # Start with the known good solution: {63, ..., 128}, size 66
    best_set = set(range(63, 129))
    best_size = len(best_set)
    
    # Try to improve via local search
    # Strategy: try adding elements from {1..62} and removing minimal conflicts
    
    def count_violations(S):
        """Count number of violating triples in set S."""
        S_list = sorted(S)
        S_set = set(S_list)
        count = 0
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        count += 1
        return count
    
    def is_valid(S):
        """Check if set S has no violating triples."""
        S_set = set(S)
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True
    
    def try_add(S, x):
        """Try adding x to S. Returns (new_set, violations) if valid, else (None, 0)."""
        if x in S:
            return S, 0
        new_S = S | {x}
        # Check if adding x creates any violation
        S_list = sorted(S)
        S_set = set(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a, b = S_list[i], S_list[j]
                # Check triple (a, b, x)
                if (a+x in S_set) and (b+x in S_set) and (a+b in S_set):
                    return None, 0
                # Also need to check if a+b, a+c, b+c where one is x
                # Actually we need: for triple (a,b,x), we need a+b, a+x, b+x all in new_S
                # a+x and b+x are in new_S by construction (x is added)
                # So we just need a+b in new_S (which is in S_set since a,b in S)
                # But wait, we also need to check triples where x replaces an existing element
        # More careful check:
        new_S_set = set(new_S)
        new_S_list = sorted(new_S)
        n = len(new_S_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = new_S_list[i], new_S_list[j], new_S_list[k]
                    if (a+b in new_S_set) and (a+c in new_S_set) and (b+c in new_S_set):
                        return None, 0
        return new_S, 0
    
    def find_min_removals(S, x):
        """Find minimum elements to remove from S to make S∪{x} valid."""
        if x in S:
            return set()
        new_S = S | {x}
        new_S_set = set(new_S)
        new_S_list = sorted(new_S)
        n = len(new_S_list)
        
        # Find all violating triples
        violations = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = new_S_list[i], new_S_list[j], new_S_list[k]
                    if (a+b in new_S_set) and (a+c in new_S_set) and (b+c in new_S_set):
                        violations.append((a, b, c))
        
        if not violations:
            return set()
        
        # Greedy: find minimum set of elements to remove to break all violations
        # Each violation involves 3 elements; removing any one breaks it
        # This is a hitting set problem; use greedy approximation
        # Sort violations by number of elements that appear in multiple violations
        from collections import Counter
        elem_count = Counter()
        for v in violations:
            for e in v:
                elem_count[e] += 1
        
        removals = set()
        remaining_violations = set(violations)
        
        while remaining_violations and len(removals) < 3:
            # Find element that appears in most remaining violations
            best_elem = None
            best_count = 0
            for e in elem_count:
                if e in removals:
                    continue
                cnt = sum(1 for v in remaining_violations if e in v)
                if cnt > best_count:
                    best_count = cnt
                    best_elem = e
            if best_elem is None or best_count == 0:
                break
            removals.add(best_elem)
            remaining_violations = {v for v in remaining_violations if best_elem not in v}
        
        # Verify
        test_S = new_S - removals
        if is_valid(test_S):
            return removals
        return None
    
    # Local search: try to add elements one by one
    candidates = list(range(1, 63))
    random.seed(42)
    
    for _ in range(50):  # Try up to 50 improvement rounds
        if time.time() > deadline - 5:
            break
        
        improved = False
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if x in best_set:
                continue
            
            # Try direct addition
            new_S, _ = try_add(best_set, x)
            if new_S is not None and len(new_S) > best_size:
                best_set = new_S
                best_size = len(best_set)
                improved = True
                continue
            
            # Try adding x with minimal removals
            removals = find_min_removals(best_set, x)
            if removals is not None and len(best_set) - len(removals) + 1 > best_size:
                new_S = (best_set - removals) | {x}
                if is_valid(new_S):
                    best_set = new_S
                    best_size = len(best_set)
                    improved = True
        
        if not improved:
            # Try random restarts with different starting points
            # Try removing some elements and adding others
            break
    
    # Also try a different initial structure
    # Try sets of form {k, k+1, ..., 128} for various k
    for k in range(1, 70):
        if time.time() > deadline - 5:
            break
        S = set(range(k, 129))
        if is_valid(S) and len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Try complement-based approaches
    # Try odd numbers only
    S_odd = set(range(1, 129, 2))
    if is_valid(S_odd) and len(S_odd) > best_size:
        best_set = S_odd
        best_size = len(S_odd)
    
    # Try even numbers only
    S_even = set(range(2, 129, 2))
    if is_valid(S_even) and len(S_even) > best_size:
        best_set = S_even
        best_size = len(S_even)
    
    # Try numbers > 64
    S_gt64 = set(range(65, 129))
    if is_valid(S_gt64) and len(S_gt64) > best_size:
        best_set = S_gt64
        best_size = len(S_gt64)
    
    # Try numbers in [40, 128]
    S_40 = set(range(40, 129))
    if is_valid(S_40) and len(S_40) > best_size:
        best_set = S_40
        best_size = len(S_40)
    
    # Try numbers in [32, 128]
    S_32 = set(range(32, 129))
    if is_valid(S_32) and len(S_32) > best_size:
        best_set = S_32
        best_size = len(S_32)
    
    # More aggressive local search with swaps
    if time.time() < deadline - 10:
        current_set = set(best_set)
        current_size = best_size
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Try a random swap: remove one element, add another
            if not current_set:
                break
            
            to_remove = random.choice(list(current_set))
            to_add = random.choice([x for x in range(1, 129) if x not in current_set])
            
            new_set = (current_set - {to_remove}) | {to_add}
            if is_valid(new_set) and len(new_set) > current_size:
                current_set = new_set
                current_size = len(new_set)
                if current_size > best_size:
                    best_set = set(current_set)
                    best_size = current_size
    
    # Final verification
    if not is_valid(best_set):
        # Fallback to known good solution
        best_set = set(range(63, 129))
        best_size = 66
    
    return list(best_set)

# EVOLVE-BLOCK-END
