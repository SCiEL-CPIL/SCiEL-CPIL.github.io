"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    n = 256
    max_elements = 64
    
    def is_valid_sidon(lst):
        """Check if lst is a Sidon set (pair sums including a+a are unique)."""
        sums = set()
        for i, a in enumerate(lst):
            for b in lst[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_diffs(lst):
        """Get nonzero differences of lst."""
        return set(b - a for a in lst for b in lst if a != b)
    
    def greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering.
        Returns (setA, setB) as sorted lists.
        """
        setA = []
        setB = []
        diffsA = set()
        diffsB = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Try adding to setA
            can_add_A = True
            if len(setA) < max_elements:
                # Check Sidon condition for setA + [x]
                new_sums = set()
                for a in setA:
                    s = a + x
                    if s in new_sums:
                        can_add_A = False
                        break
                    new_sums.add(s)
                if can_add_A:
                    # Check x + x
                    s_xx = 2 * x
                    if s_xx in new_sums:
                        can_add_A = False
                    else:
                        # Check existing sums in setA
                        for i, a in enumerate(setA):
                            for b in setA[i:]:
                                if a + b == s_xx:
                                    can_add_A = False
                                    break
                            if not can_add_A:
                                break
                
                # Check difference disjointness
                if can_add_A:
                    new_diffs = set()
                    for a in setA:
                        d = x - a
                        new_diffs.add(d)
                        d2 = a - x
                        new_diffs.add(d2)
                    if new_diffs & diffsB:
                        can_add_A = False
            
            # Try adding to setB
            can_add_B = True
            if len(setB) < max_elements:
                new_sums = set()
                for b in setB:
                    s = b + x
                    if s in new_sums:
                        can_add_B = False
                        break
                    new_sums.add(s)
                if can_add_B:
                    s_xx = 2 * x
                    if s_xx in new_sums:
                        can_add_B = False
                    else:
                        for i, b in enumerate(setB):
                            for c in setB[i:]:
                                if b + c == s_xx:
                                    can_add_B = False
                                    break
                            if not can_add_B:
                                break
                
                if can_add_B:
                    new_diffs = set()
                    for b in setB:
                        d = x - b
                        new_diffs.add(d)
                        d2 = b - x
                        new_diffs.add(d2)
                    if new_diffs & diffsA:
                        can_add_B = False
            
            # Decide where to add
            if can_add_A and not can_add_B:
                setA.append(x)
                for a in setA[:-1]:
                    diffsA.add(x - a)
                    diffsA.add(a - x)
            elif can_add_B and not can_add_A:
                setB.append(x)
                for b in setB[:-1]:
                    diffsB.add(x - b)
                    diffsB.add(b - x)
            elif can_add_A and can_add_B:
                # Add to the smaller set to balance
                if len(setA) <= len(setB):
                    setA.append(x)
                    for a in setA[:-1]:
                        diffsA.add(x - a)
                        diffsA.add(a - x)
                else:
                    setB.append(x)
                    for b in setB[:-1]:
                        diffsB.add(x - b)
                        diffsB.add(b - x)
        
        return sorted(setA), sorted(setB)
    
    def local_search(setA, setB, deadline):
        """
        Local search: try swapping elements to improve total count.
        """
        setA = sorted(setA)
        setB = sorted(setB)
        diffsA = get_diffs(setA)
        diffsB = get_diffs(setB)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements not in either set
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x in setA or x in setB:
                    continue
                
                # Try adding to A
                can_add_A = True
                if len(setA) < max_elements:
                    # Check Sidon
                    for a in setA:
                        if (a + x) in {a1 + a2 for a1 in setA for a2 in setA if a1 <= a2} | {2*a for a in setA}:
                            can_add_A = False
                            break
                    if can_add_A:
                        new_diffs = set(x - a for a in setA) | set(a - x for a in setA)
                        if new_diffs & diffsB:
                            can_add_A = False
                
                # Try adding to B
                can_add_B = True
                if len(setB) < max_elements:
                    for b in setB:
                        if (b + x) in {b1 + b2 for b1 in setB for b2 in setB if b1 <= b2} | {2*b for b in setB}:
                            can_add_B = False
                            break
                    if can_add_B:
                        new_diffs = set(x - b for b in setB) | set(b - x for b in setB)
                        if new_diffs & diffsA:
                            can_add_B = False
                
                if can_add_A:
                    setA.append(x)
                    setA.sort()
                    diffsA = get_diffs(setA)
                    improved = True
                elif can_add_B:
                    setB.append(x)
                    setB.sort()
                    diffsB = get_diffs(setB)
                    improved = True
        
        return setA, setB
    
    # Try multiple random orderings
    best_A, best_B = [], []
    best_size = 0
    
    base_order = list(range(1, n + 1))
    
    # First, try natural order
    A, B = greedy_one_ordering(base_order, deadline)
    A, B = local_search(A, B, deadline)
    best_A, best_B = A, B
    best_size = len(A) + len(B)
    
    # Try several random orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        order = base_order[:]
        random.shuffle(order)
        A, B = greedy_one_ordering(order, deadline)
        A, B = local_search(A, B, deadline)
        if len(A) + len(B) > best_size:
            best_A, best_B = A, B
            best_size = len(A) + len(B)
    
    # Final local search on best result
    best_A, best_B = local_search(best_A, best_B, deadline)
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
