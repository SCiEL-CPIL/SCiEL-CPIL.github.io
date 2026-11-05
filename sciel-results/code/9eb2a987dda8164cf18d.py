"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    max_size = 64
    
    def _validate_sidon(aset):
        """Check if aset is a valid Sidon set (all pair sums including a+a are unique)."""
        sums = set()
        for i, a in enumerate(aset):
            for j in range(i, len(aset)):
                s = a + aset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def _get_diffs(aset):
        """Get all nonzero differences from aset."""
        diffs = set()
        for i in range(len(aset)):
            for j in range(len(aset)):
                if i != j:
                    d = aset[i] - aset[j]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def _greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering.
        Returns (setA, setB) as sorted lists.
        """
        setA = []
        setB = []
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Try adding x to setA
            can_add_A = True
            new_sumsA = set(sumsA)
            for a in setA:
                s = a + x
                if s in new_sumsA:
                    can_add_A = False
                    break
                new_sumsA.add(s)
            s2 = x + x
            if can_add_A and s2 in new_sumsA:
                can_add_A = False
            else:
                new_sumsA.add(s2)
            
            if can_add_A:
                new_diffsA = set(diffsA)
                for a in setA:
                    d1 = x - a
                    d2 = a - x
                    if d1 in new_diffsA or d2 in new_diffsA:
                        can_add_A = False
                        break
                    new_diffsA.add(d1)
                    new_diffsA.add(d2)
                
                if can_add_A and new_diffsA & diffsB:
                    can_add_A = False
            
            if can_add_A:
                setA.append(x)
                sumsA = new_sumsA
                diffsA = new_diffsA
                continue
            
            # Try adding x to setB
            can_add_B = True
            new_sumsB = set(sumsB)
            for b in setB:
                s = b + x
                if s in new_sumsB:
                    can_add_B = False
                    break
                new_sumsB.add(s)
            s2 = x + x
            if can_add_B and s2 in new_sumsB:
                can_add_B = False
            else:
                new_sumsB.add(s2)
            
            if can_add_B:
                new_diffsB = set(diffsB)
                for b in setB:
                    d1 = x - b
                    d2 = b - x
                    if d1 in new_diffsB or d2 in new_diffsB:
                        can_add_B = False
                        break
                    new_diffsB.add(d1)
                    new_diffsB.add(d2)
                
                if can_add_B and new_diffsB & diffsA:
                    can_add_B = False
            
            if can_add_B:
                setB.append(x)
                sumsB = new_sumsB
                diffsB = new_diffsB
        
        return setA, setB
    
    def _local_search(setA, setB, deadline):
        """
        Try to improve by swapping elements or adding new ones.
        Returns improved (setA, setB).
        """
        setA = list(setA)
        setB = list(setB)
        
        # Precompute state
        def _rebuild_state(aset):
            sums = set()
            for i, a in enumerate(aset):
                for j in range(i, len(aset)):
                    sums.add(a + aset[j])
            diffs = set()
            for i in range(len(aset)):
                for j in range(len(aset)):
                    if i != j:
                        diffs.add(aset[i] - aset[j])
            return sums, diffs
        
        sumsA, diffsA = _rebuild_state(setA)
        sumsB, diffsB = _rebuild_state(setB)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            all_elements = list(range(1, n + 1))
            
            # Try adding new elements
            for x in all_elements:
                if x in setA or x in setB:
                    continue
                
                # Try adding to A
                can_add_A = True
                for a in setA:
                    if (a + x) in sumsA or (x + x) in sumsA:
                        can_add_A = False
                        break
                if can_add_A:
                    for a in setA:
                        if (x - a) in diffsA or (a - x) in diffsA:
                            can_add_A = False
                            break
                if can_add_A and diffsA & diffsB:
                    # Check new diffs against B's diffs
                    for a in setA:
                        if (x - a) in diffsB or (a - x) in diffsB:
                            can_add_A = False
                            break
                
                if can_add_A:
                    setA.append(x)
                    sumsA, diffsA = _rebuild_state(setA)
                    improved = True
                    break
                
                # Try adding to B
                can_add_B = True
                for b in setB:
                    if (b + x) in sumsB or (x + x) in sumsB:
                        can_add_B = False
                        break
                if can_add_B:
                    for b in setB:
                        if (x - b) in diffsB or (b - x) in diffsB:
                            can_add_B = False
                            break
                if can_add_B:
                    for b in setB:
                        if (x - b) in diffsA or (b - x) in diffsA:
                            can_add_B = False
                            break
                
                if can_add_B:
                    setB.append(x)
                    sumsB, diffsB = _rebuild_state(setB)
                    improved = True
                    break
        
        return setA, setB
    
    best_size = 0
    best_result = ([], [])
    
    # Multiple random restarts with different orderings
    while time.time() < deadline:
        # Create a random ordering
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        setA, setB = _greedy_one_ordering(order, deadline)
        current_size = len(setA) + len(setB)
        
        if current_size > best_size:
            best_size = current_size
            best_result = (list(setA), list(setB))
        
        # Try local search to improve
        if time.time() < deadline - 5:
            setA_ls, setB_ls = _local_search(setA, setB, deadline)
            ls_size = len(setA_ls) + len(setB_ls)
            if ls_size > best_size:
                best_size = ls_size
                best_result = (list(setA_ls), list(setB_ls))
    
    # Ensure nonempty
    if not best_result[0]:
        best_result[0] = [1]
    if not best_result[1]:
        best_result[1] = [2]
    
    return [sorted(best_result[0]), sorted(best_result[1])]
# EVOLVE-BLOCK-END
