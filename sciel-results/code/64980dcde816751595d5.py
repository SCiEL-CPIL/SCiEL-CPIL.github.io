"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    
    def _compute_sums(elems):
        """Compute all pair sums including a+a."""
        s = set()
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s.add(elems[i] + elems[j])
        return s
    
    def _compute_diffs(elems):
        """Compute all nonzero differences within the set."""
        d = set()
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                diff = abs(elems[i] - elems[j])
                if diff > 0:
                    d.add(diff)
        return d
    
    def _is_valid_sidon(elems):
        """Check if elems forms a Sidon set (unique pair sums)."""
        sums = _compute_sums(elems)
        return len(sums) == len(elems) * (len(elems) + 1) // 2
    
    def _can_add_to_sidon(x, elems, sums):
        """Check if x can be added to maintain Sidon property."""
        for a in elems:
            if a + x in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def _greedy_build_sidon(order, excluded_diffs, deadline):
        """
        Greedily build a Sidon set, avoiding differences in excluded_diffs.
        Returns the constructed set.
        """
        elems = []
        sums = set()
        diffs = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Check Sidon constraint
            if not _can_add_to_sidon(x, elems, sums):
                continue
            
            # Check difference constraint: new diffs must not conflict with excluded_diffs
            new_diffs_conflict = False
            for a in elems:
                d = abs(a - x)
                if d > 0 and d in excluded_diffs:
                    new_diffs_conflict = True
                    break
            
            if new_diffs_conflict:
                continue
            
            # Add x
            for a in elems:
                sums.add(a + x)
                d = abs(a - x)
                if d > 0:
                    diffs.add(d)
            sums.add(2 * x)
            elems.append(x)
        
        return sorted(elems)
    
    def _local_search_two_sets(setA, setB, deadline, max_iters=2000):
        """
        Local search: try swapping elements between sets or adding/removing.
        """
        setA = list(setA)
        setB = list(setB)
        
        def _validate(A, B):
            if not A or not B:
                return False
            if not _is_valid_sidon(A):
                return False
            if not _is_valid_sidon(B):
                return False
            diffA = _compute_diffs(A)
            diffB = _compute_diffs(B)
            if diffA & diffB:
                return False
            return True
        
        def _size(A, B):
            return len(A) + len(B)
        
        best_A, best_B = list(setA), list(setB)
        best_size = _size(best_A, best_B)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random move: try to add an element to A or B, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try adding a random element to A
                x = random.randint(1, n)
                if x not in setA:
                    new_A = setA + [x]
                    new_A.sort()
                    if _validate(new_A, setB):
                        setA = new_A
                        if _size(setA, setB) > best_size:
                            best_A, best_B = list(setA), list(setB)
                            best_size = _size(best_A, best_B)
            
            elif move_type < 0.6:
                # Try adding a random element to B
                x = random.randint(1, n)
                if x not in setB:
                    new_B = setB + [x]
                    new_B.sort()
                    if _validate(setA, new_B):
                        setB = new_B
                        if _size(setA, setB) > best_size:
                            best_A, best_B = list(setA), list(setB)
                            best_size = _size(best_A, best_B)
            
            elif move_type < 0.8:
                # Try swapping: remove from A, add to B
                if setA:
                    idx = random.randint(0, len(setA) - 1)
                    removed = setA[idx]
                    new_A = setA[:idx] + setA[idx+1:]
                    x = random.randint(1, n)
                    if x not in new_A:
                        new_B = setB + [x]
                        new_B.sort()
                        if _validate(new_A, new_B):
                            setA, setB = new_A, new_B
                            if _size(setA, setB) > best_size:
                                best_A, best_B = list(setA), list(setB)
                                best_size = _size(best_A, best_B)
            
            else:
                # Try swapping: remove from B, add to A
                if setB:
                    idx = random.randint(0, len(setB) - 1)
                    removed = setB[idx]
                    new_B = setB[:idx] + setB[idx+1:]
                    x = random.randint(1, n)
                    if x not in new_B:
                        new_A = setA + [x]
                        new_A.sort()
                        if _validate(new_A, new_B):
                            setA, setB = new_A, new_B
                            if _size(setA, setB) > best_size:
                                best_A, best_B = list(setA), list(setB)
                                best_size = _size(best_A, best_B)
        
        return best_A, best_B
    
    # Strategy: Build A first with a good structure, then build B avoiding A's differences
    best_sets = ([1], [2])
    best_size = 2
    
    rng = random.Random(42)
    
    for restart in range(300):
        if time.time() > deadline:
            break
        
        # Build A greedily
        order_A = list(range(1, n + 1))
        rng.shuffle(order_A)
        setA = _greedy_build_sidon(order_A, set(), deadline)
        
        if not setA:
            continue
        
        # Compute A's differences
        diffA = _compute_diffs(setA)
        
        # Build B avoiding A's differences
        order_B = list(range(1, n + 1))
        rng.shuffle(order_B)
        setB = _greedy_build_sidon(order_B, diffA, deadline)
        
        if not setB:
            continue
        
        current_size = len(setA) + len(setB)
        
        # Local search to improve
        if current_size > best_size or (current_size >= best_size and restart > 50):
            setA_ls, setB_ls = _local_search_two_sets(setA, setB, deadline, max_iters=500)
            ls_size = len(setA_ls) + len(setB_ls)
            if ls_size > best_size:
                best_sets = (setA_ls, setB_ls)
                best_size = ls_size
        elif current_size > best_size:
            best_sets = (setA, setB)
            best_size = current_size
    
    return [best_sets[0], best_sets[1]]
# EVOLVE-BLOCK-END
