# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from itertools import combinations_with_replacement, combinations

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 80  # safety margin before 90s kill
    
    def is_valid_sidon(aset):
        """Check if aset is a valid Sidon set (unique pair sums including a+a)."""
        if len(aset) <= 1:
            return True
        sums = set()
        for a in aset:
            for b in aset:
                if a <= b:
                    s = a + b
                    if s in sums:
                        return False
                    sums.add(s)
        return True
    
    def get_diffs(aset):
        """Get nonzero differences of aset."""
        diffs = set()
        for a in aset:
            for b in aset:
                if a != b:
                    diffs.add(b - a)
        return diffs
    
    def greedy_build(order, side, existing_other, max_size, deadline):
        """
        Greedily build a Sidon set from candidates in 'order',
        given existing_other's difference set.
        Returns the built set.
        """
        aset = []
        used_sums = set()
        used_diffs = set()
        other_diffs = get_diffs(existing_other)
        
        for x in order:
            if time.time() > deadline:
                break
            if len(aset) >= max_size:
                break
            
            # Check if adding x keeps Sidon property
            # New sums: x + a for all a in aset, and x + x
            valid = True
            new_sums = set()
            for a in aset:
                s = x + a
                if s in used_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                s2 = x + x
                if s2 in used_sums or s2 in new_sums:
                    valid = False
                else:
                    new_sums.add(s2)
            
            if not valid:
                continue
            
            # Check if adding x creates difference conflicts
            # New differences: x - a and a - x for all a in aset
            new_diffs = set()
            for a in aset:
                d1 = x - a
                d2 = a - x
                if d1 in used_diffs or d1 in other_diffs or d1 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d1)
                if d2 in used_diffs or d2 in other_diffs or d2 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d2)
            
            if not valid:
                continue
            
            # Accept x
            aset.append(x)
            used_sums.update(new_sums)
            used_diffs.update(new_diffs)
        
        return aset
    
    def greedy_build_both(order_a, order_b, deadline):
        """
        Build A from order_a and B from order_b, ensuring disjoint differences.
        We build A first, then B avoiding A's differences.
        """
        A = []
        used_sums_a = set()
        used_diffs_a = set()
        
        for x in order_a:
            if time.time() > deadline:
                break
            if len(A) >= max_size:
                break
            
            valid = True
            new_sums = set()
            for a in A:
                s = x + a
                if s in used_sums_a or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                s2 = x + x
                if s2 in used_sums_a or s2 in new_sums:
                    valid = False
                else:
                    new_sums.add(s2)
            
            if not valid:
                continue
            
            new_diffs = set()
            for a in A:
                d1 = x - a
                d2 = a - x
                if d1 in used_diffs_a or d1 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d1)
                if d2 in used_diffs_a or d2 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d2)
            
            if not valid:
                continue
            
            A.append(x)
            used_sums_a.update(new_sums)
            used_diffs_a.update(new_diffs)
        
        # Now build B avoiding A's differences
        B = []
        used_sums_b = set()
        used_diffs_b = set()
        
        for x in order_b:
            if time.time() > deadline:
                break
            if len(B) >= max_size:
                break
            
            valid = True
            new_sums = set()
            for b in B:
                s = x + b
                if s in used_sums_b or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                s2 = x + x
                if s2 in used_sums_b or s2 in new_sums:
                    valid = False
                else:
                    new_sums.add(s2)
            
            if not valid:
                continue
            
            new_diffs = set()
            for b in B:
                d1 = x - b
                d2 = b - x
                if d1 in used_diffs_b or d1 in used_diffs_a or d1 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d1)
                if d2 in used_diffs_b or d2 in used_diffs_a or d2 in new_diffs:
                    valid = False
                    break
                new_diffs.add(d2)
            
            if not valid:
                continue
            
            B.append(x)
            used_sums_b.update(new_sums)
            used_diffs_b.update(new_diffs)
        
        return A, B
    
    def verify(A, B):
        """Verify both sets are valid Sidon sets with disjoint nonzero differences."""
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        
        # Check Sidon property for A
        sums_a = set()
        for a in A:
            for b in A:
                if a <= b:
                    s = a + b
                    if s in sums_a:
                        return False
                    sums_a.add(s)
        
        # Check Sidon property for B
        sums_b = set()
        for b in B:
            for c in B:
                if b <= c:
                    s = b + c
                    if s in sums_b:
                        return False
                    sums_b.add(s)
        
        # Check disjoint differences
        diffs_a = set()
        for a in A:
            for b in A:
                if a != b:
                    d = b - a
                    diffs_a.add(d)
        
        diffs_b = set()
        for b in B:
            for c in B:
                if b != c:
                    d = c - b
                    if d in diffs_a:
                        return False
                    diffs_b.add(d)
        
        return True
    
    # Try multiple random orderings
    best_score = 0
    best_A, best_B = None, None
    
    # Initial deterministic attempt
    order = list(range(1, n + 1))
    A, B = greedy_build_both(order, order, deadline)
    if verify(A, B):
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]
    
    # Random restarts
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline - 5:
        attempt += 1
        
        # Create random orders
        order_a = list(range(1, n + 1))
        order_b = list(range(1, n + 1))
        rng.shuffle(order_a)
        rng.shuffle(order_b)
        
        A, B = greedy_build_both(order_a, order_b, deadline)
        
        if verify(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A, best_B = A[:], B[:]
                # If we hit the theoretical max, stop
                if best_score >= 2 * max_size:
                    break
    
    # If no valid solution found, return a trivial one
    if best_A is None or best_B is None:
        # Minimal valid solution
        A = [1, 2]
        B = [3, 4]
        if not verify(A, B):
            A = [1]
            B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
