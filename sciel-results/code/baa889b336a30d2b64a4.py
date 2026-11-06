import random
import sys
from itertools import combinations, combinations_with_replacement
import heapq
import time

def construct():
    n = 256
    max_size = 64
    
    def can_add_to_side(S, sums_S, diffs_other, x):
        """
        Check if x can be added to side S.
        S: list of elements
        sums_S: set of all pair sums in S (including doubles)
        diffs_other: set of nonzero differences in the other side
        Returns: (bool, new_sums, new_diffs)
        """
        # Check Sidon: new sums must not be in sums_S
        # New sums: x+x, x+s for s in S
        new_sums = {x + x}
        for s in S:
            new_sums.add(x + s)
        
        if new_sums & sums_S:
            return False, None, None
        
        # Check Diffs: new diffs must not be in diffs_other
        # New diffs: x-s, s-x for s in S
        new_diffs = set()
        for s in S:
            new_diffs.add(x - s)
            new_diffs.add(s - x)
            
        if new_diffs & diffs_other:
            return False, None, None
        
        return True, new_sums, new_diffs

    best_total = 0
    best_A = []
    best_B = []
    
    start_time = time.time()
    timeout = 55.0
    
    for trial in range(5000):
        if time.time() - start_time > timeout:
            break
            
        random.seed(trial)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in candidates:
            # Try A
            if len(A) < max_size:
                ok, ns, nd = can_add_to_side(A, sums_A, diffs_B, x)
                if ok:
                    A.append(x)
                    sums_A.update(ns)
                    diffs_A.update(nd)
                    continue
            
            # Try B
            if len(B) < max_size:
                ok, ns, nd = can_add_to_side(B, sums_B, diffs_A, x)
                if ok:
                    B.append(x)
                    sums_B.update(ns)
                    diffs_B.update(nd)
                    continue

        # Local Search: Try to add any remaining candidate
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            for x in range(1, n + 1):
                if x in A or x in B:
                    continue
                
                # Try A
                if len(A) < max_size:
                    ok, ns, nd = can_add_to_side(A, sums_A, diffs_B, x)
                    if ok:
                        A.append(x)
                        sums_A.update(ns)
                        diffs_A.update(nd)
                        improved = True
                        break
                
                # Try B
                if len(B) < max_size:
                    ok, ns, nd = can_add_to_side(B, sums_B, diffs_A, x)
                    if ok:
                        B.append(x)
                        sums_B.update(ns)
                        diffs_B.update(nd)
                        improved = True
                        break
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]

    return [best_A, best_B]
