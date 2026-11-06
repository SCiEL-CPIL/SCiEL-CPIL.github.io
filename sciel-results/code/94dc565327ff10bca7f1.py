"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = 256
    max_size = 64
    
    # Precompute all possible pairs for efficiency
    # For a set S, we need:
    # 1. All pair sums (a+b for a<=b in S) to be unique
    # 2. All nonzero differences (b-a for a<b in S) to be disjoint between A and B
    
    def get_sums_and_diffs(S):
        """Compute pair sums (with replacement) and nonzero differences for set S."""
        sums = set()
        diffs = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return sums, diffs
    
    def can_add_to_set(element, S, S_sums, S_diffs, other_diffs):
        """Check if element can be added to S without violating constraints."""
        # Check new sums
        for x in S:
            if element + x in S_sums:
                return False
        if element + element in S_sums:
            return False
        
        # Check new differences don't conflict with other_diffs
        for x in S:
            d = abs(element - x)
            if d in other_diffs:
                return False
        
        return True
    
    def add_to_set(element, S, S_sums, S_diffs):
        """Add element to S and update sums and diffs."""
        S_new = S + [element]
        S_new.sort()
        
        # Update sums
        for x in S:
            S_sums.add(element + x)
        S_sums.add(element + element)
        
        # Update diffs
        for x in S:
            S_diffs.add(abs(element - x))
        
        return S_new, S_sums, S_diffs
    
    # Strategy: Build both sets simultaneously using a greedy approach
    # with backtracking and multiple restarts
    
    best_result = ([1], [2])
    best_size = 2
    
    # Try multiple random orderings
    for trial in range(200):
        if time.time() - start_time > time_limit:
            break
        
        if trial == 0:
            order = list(range(1, n + 1))
        else:
            order = list(range(1, n + 1))
            random.shuffle(order)
        
        A = []
        B = []
        A_sums = set()
        B_sums = set()
        A_diffs = set()
        B_diffs = set()
        
        for x in order:
            if time.time() - start_time > time_limit:
                break
            
            # Try adding to A
            can_add_A = True
            for a in A:
                if x + a in A_sums:
                    can_add_A = False
                    break
                if abs(x - a) in B_diffs:
                    can_add_A = False
                    break
            if can_add_A and x + x in A_sums:
                can_add_A = False
            
            if can_add_A:
                # Update A
                for a in A:
                    A_sums.add(x + a)
                    A_diffs.add(abs(x - a))
                A_sums.add(x + x)
                A.append(x)
                A.sort()
                continue
            
            # Try adding to B
            can_add_B = True
            for b in B:
                if x + b in B_sums:
                    can_add_B = False
                    break
                if abs(x - b) in A_diffs:
                    can_add_B = False
                    break
            if can_add_B and x + x in B_sums:
                can_add_B = False
            
            if can_add_B:
                # Update B
                for b in B:
                    B_sums.add(x + b)
                    B_diffs.add(abs(x - b))
                B_sums.add(x + x)
                B.append(x)
                B.sort()
                continue
        
        # Check if we have valid non-empty sets
        if len(A) > 0 and len(B) > 0:
            total_size = len(A) + len(B)
            if total_size > best_size:
                best_size = total_size
                best_result = (A[:], B[:])
    
    return list(best_result)
# EVOLVE-BLOCK-END
