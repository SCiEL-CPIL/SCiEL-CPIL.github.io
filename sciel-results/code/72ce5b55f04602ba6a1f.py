"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations_with_replacement

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 80  # safety margin before 90s limit
    
    best_total = 0
    best_A = [1]
    best_B = [2]
    
    # Precompute all possible pairs for difference checking
    # We'll use a greedy randomized approach with multiple restarts
    
    def check_sidon(aset):
        """Check if aset is a Sidon set (pair sums unique including a+a)."""
        sums = set()
        for i in range(len(aset)):
            for j in range(i, len(aset)):
                s = aset[i] + aset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_diffs(aset):
        """Get set of nonzero differences."""
        diffs = set()
        for i in range(len(aset)):
            for j in range(len(aset)):
                if i != j:
                    d = abs(aset[i] - aset[j])
                    if d > 0:
                        diffs.add(d)
        return diffs
    
    def can_add(aset, x, other_diffs):
        """Check if x can be added to aset without violating Sidon or diff constraints."""
        # Check Sidon condition
        for a in aset:
            s = a + x
            for b in aset:
                if b != a:
                    if s == a + b or s == b + x:
                        # More careful check needed
                        pass
        
        # Better: just check all pair sums
        new_sums = set()
        for i in range(len(aset)):
            for j in range(i, len(aset)):
                s = aset[i] + aset[j]
                new_sums.add(s)
        
        # Check new pairs with x
        for a in aset:
            s = a + x
            if s in new_sums:
                return False
            new_sums.add(s)
        s = x + x
        if s in new_sums:
            return False
        
        # Check difference constraints
        new_diffs = set()
        for a in aset:
            d = abs(a - x)
            if d > 0:
                new_diffs.add(d)
        
        if new_diffs & other_diffs:
            return False
        
        return True
    
    def greedy_fill(elements_order, side_a_first=True):
        """Greedy fill using given element order."""
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        for x in elements_order:
            if len(A) >= max_size and len(B) >= max_size:
                break
            
            # Try adding to A
            if len(A) < max_size:
                if can_add(A, x, diffs_B):
                    # Check Sidon
                    valid = True
                    sums = set()
                    for i in range(len(A)):
                        for j in range(i, len(A)):
                            s = A[i] + A[j]
                            sums.add(s)
                    for a in A:
                        s = a + x
                        if s in sums:
                            valid = False
                            break
                    if valid:
                        if x + x in sums:
                            valid = False
                    if valid:
                        # Update diffs
                        for a in A:
                            d = abs(a - x)
                            if d > 0:
                                diffs_A.add(d)
                        A.append(x)
                        continue
            
            # Try adding to B
            if len(B) < max_size:
                if can_add(B, x, diffs_A):
                    valid = True
                    sums = set()
                    for i in range(len(B)):
                        for j in range(i, len(B)):
                            s = B[i] + B[j]
                            sums.add(s)
                    for b in B:
                        s = b + x
                        if s in sums:
                            valid = False
                            break
                    if valid:
                        if x + x in sums:
                            valid = False
                    if valid:
                        for b in B:
                            d = abs(b - x)
                            if d > 0:
                                diffs_B.add(d)
                        B.append(x)
        
        return A, B
    
    # Multiple randomized passes
    elements = list(range(1, n + 1))
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Shuffle elements
        random.shuffle(elements)
        
        A, B = greedy_fill(elements)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Final validation
    def validate(A, B):
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        
        # Check disjoint nonzero differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = abs(A[i] - A[j])
                    if d > 0:
                        diffs_A.add(d)
        
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = abs(B[i] - B[j])
                    if d > 0:
                        diffs_B.add(d)
        
        if diffs_A & diffs_B:
            return False
        
        return True
    
    if not validate(best_A, best_B):
        # Fallback to simple valid solution
        best_A = [1, 2, 4, 8, 16, 32, 64, 128]
        best_B = [3, 6, 12, 24, 48, 96]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
