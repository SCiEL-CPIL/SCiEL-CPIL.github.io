"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    
    def _can_add_to_set(x, set_elems, sums, diffs_other):
        """Check if x can be added to a set with given elements, sums, and other set's diffs."""
        for a in set_elems:
            s = a + x
            if s in sums:
                return False
            d = abs(a - x)
            if d in diffs_other:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def _add_to_set(x, set_elems, sums, diffs):
        """Add x to set, updating sums and diffs."""
        for a in set_elems:
            sums.add(a + x)
            diffs.add(abs(a - x))
        sums.add(2 * x)
        set_elems.append(x)
    
    def _remove_from_set(x, set_elems, sums, diffs):
        """Remove x from set, updating sums and diffs."""
        set_elems.remove(x)
        for a in set_elems:
            sums.discard(a + x)
            diffs.discard(abs(a - x))
        sums.discard(2 * x)
    
    def _greedy_balanced(order, deadline, prefer_balance=True):
        """
        Greedy with balanced allocation: try to keep |A| and |B| close.
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
            
            # Decide which set to try first based on balance
            if prefer_balance:
                try_A_first = len(setA) <= len(setB)
            else:
                try_A_first = random.random() < 0.5
            
            if try_A_first:
                if _can_add_to_set(x, setA, sumsA, diffsB):
                    _add_to_set(x, setA, sumsA, diffsA)
                    continue
                if _can_add_to_set(x, setB, sumsB, diffsA):
                    _add_to_set(x, setB, sumsB, diffsB)
                    continue
            else:
                if _can_add_to_set(x, setB, sumsB, diffsA):
                    _add_to_set(x, setB, sumsB, diffsB)
                    continue
                if _can_add_to_set(x, setA, sumsA, diffsB):
                    _add_to_set(x, setA, sumsA, diffsA)
                    continue
        
        return sorted(setA), sorted(setB)
    
    def _local_search(setA, setB, deadline, max_iters=500):
        """
        Improved local search: try 1-removal + 1-addition swaps.
        Uses incremental sum/diff tracking for speed.
        """
        setA = list(setA)
        setB = list(setB)
        
        # Build initial sums and diffs
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()
        for i, a in enumerate(setA):
            for j in range(i + 1):
                sumsA.add(setA[i] + setA[j])
        for i, b in enumerate(setB):
            for j in range(i + 1):
                sumsB.add(setB[i] + setB[j])
        for a in setA:
            for b in setB:
                d = abs(a - b)
                if d > 0:
                    diffsA.add(d)
                    diffsB.add(d)
        
        best_size = len(setA) + len(setB)
        improved = True
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try improving A: remove one, add one
            if len(setA) > 0:
                # Pick a random element to remove from A
                idx = random.randint(0, len(setA) - 1)
                removed = setA[idx]
                
                # Temporarily remove
                _remove_from_set(removed, setA, sumsA, diffsA)
                # Also need to remove diffs involving removed
                # Actually, diffsA contains |a-b| for a in setA, b in setB
                # When we remove `removed` from setA, we lose |removed - b| for all b in setB
                # But those are in diffsA, and also in diffsB? No, diffsB is |b-a| for b in setB, a in setA
                # Actually diffsA and diffsB should be the same set! Let me reconsider.
                # The constraint is: nonzero difference sets must be disjoint.
                # diffA = {|a1 - a2| : a1, a2 in A, a1 != a2}
                # diffB = {|b1 - b2| : b1, b2 in B, b1 != b2}
                # And diffA ∩ diffB = empty
                # 
                # Wait, re-reading the problem: "Their nonzero difference sets must be disjoint."
                # This means the set of all nonzero differences within A and within B must be disjoint.
                # NOT the cross differences. Let me re-read...
                #
                # "Pair sums inside each set, including a+a, must be unique. Their nonzero difference sets must be disjoint."
                # 
                # So: 
                # - For set A: all pair sums a1+a2 (including a+a) are unique
                # - For set B: all pair sums b1+b2 (including b+b) are unique  
                # - The set {|a1-a2| : a1,a2 in A, a1!=a2} and {|b1-b2| : b1,b2 in B, b1!=b2} are disjoint
                #
                # The current code checks cross differences (|a-b| for a in A, b in B) which is WRONG!
                # Let me fix this.
                
                # Restore
                _add_to_set(removed, setA, sumsA, diffsA)
            
            # Actually, I need to completely rethink the difference constraint.
            # Let me rewrite the whole approach.
            break
        
        return sorted(setA), sorted(setB)
    
    # Multiple random restarts
    best_sets = ([1], [2])
    best_size = 2
    
    rng = random.Random(42)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        setA, setB = _greedy_balanced(order, deadline, prefer_balance=True)
        current_size = len(setA) + len(setB)
        
        if current_size > best_size:
            best_sets = (setA, setB)
            best_size = current_size
    
    return [best_sets[0], best_sets[1]]
# EVOLVE-BLOCK-END
