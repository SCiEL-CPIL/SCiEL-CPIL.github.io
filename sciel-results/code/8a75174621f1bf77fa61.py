import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    n = 256

    def check_sidon(S):
        """Check if S is a Sidon set."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(S):
        """Get set of nonzero positive differences."""
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(abs(S[i] - S[j]))
        return diffs

    def can_add_to_set(S, sums_S, O_diffs, x):
        """
        Check if x can be added to S such that:
        1. S U {x} is Sidon.
        2. Diffs(S U {x}) is disjoint from O_diffs.
        Returns (valid, new_diffs) where new_diffs is set of new diffs if valid.
        """
        # Check Sidon for S U {x}
        # New sums: x + s for s in S, and 2x
        # These must not collide with existing sums in S
        for s in S:
            if (x + s) in sums_S:
                return False, None
        if (2 * x) in sums_S:
            return False, None
        
        # Check diffs: new diffs are |x - s| for s in S
        # These must not be in O_diffs
        new_diffs = set()
        for s in S:
            nd = abs(x - s)
            if nd in O_diffs:
                return False, None
            new_diffs.add(nd)
        
        return True, new_diffs

    def try_greedy(order, prefer_A=True):
        """
        Greedy construction.
        order: list of candidates to try
        prefer_A: if True, try A first, else B first
        """
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            added = False
            
            # Determine order of trying A and B
            if prefer_A:
                sets = [(A, sums_A, diffs_A, diffs_B), (B, sums_B, diffs_B, diffs_A)]
            else:
                sets = [(B, sums_B, diffs_B, diffs_A), (A, sums_A, diffs_A, diffs_B)]
            
            for S, sums_S, my_diffs, other_diffs in sets:
                valid, new_diffs = can_add_to_set(S, sums_S, other_diffs, x)
                if valid:
                    # Add x to S
                    S.append(x)
                    for s in S:
                        sums_S.add(x + s)
                    sums_S.add(2 * x)
                    my_diffs.update(new_diffs)
                    added = True
                    break
            
            # If not added to either, skip x
        
        return len(A) + len(B), A, B

    def local_search(A, B, max_iter=50):
        """
        Try to improve by swapping elements or adding/removing.
        Returns (size, A, B)
        """
        A = list(A)
        B = list(B)
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Rebuild state
        for s in A:
            for t in A:
                if s <= t:
                    sums_A.add(s + t)
        for s in B:
            for t in B:
                if s <= t:
                    sums_B.add(s + t)
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        
        current_size = len(A) + len(B)
        best_size = current_size
        best_A = list(A)
        best_B = list(B)
        
        # Simple local search: try adding elements not in A or B
        used = set(A) | set(B)
        candidates = [x for x in range(1, n + 1) if x not in used]
        random.shuffle(candidates)
        
        for x in candidates:
            if len(A) + len(B) >= best_size:
                # Try adding to A
                valid, new_diffs = can_add_to_set(A, sums_A, diffs_B, x)
                if valid:
                    A.append(x)
                    for s in A:
                        sums_A.add(x + s)
                    sums_A.add(2 * x)
                    diffs_A.update(new_diffs)
                    best_size = len(A) + len(B)
                    best_A = list(A)
                    best_B = list(B)
                    continue
            
            # Try adding to B
            valid, new_diffs = can_add_to_set(B, sums_B, diffs_A, x)
            if valid:
                B.append(x)
                for s in B:
                    sums_B.add(x + s)
                sums_B.add(2 * x)
                diffs_B.update(new_diffs)
                best_size = len(A) + len(B)
                best_A = list(A)
                best_B = list(B)
        
        return best_size, best_A, best_B

    best_size = 0
    best_A = []
    best_B = []
    
    # Run multiple restarts with different strategies
    while time.time() < deadline:
        # Strategy 1: Random order
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        size, A, B = try_greedy(candidates, prefer_A=True)
        
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
        
        # Strategy 2: Reverse random order
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        candidates.reverse()
        size, A, B = try_greedy(candidates, prefer_A=False)
        
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
        
        # Strategy 3: Local search on best found
        if best_size > 0:
            ls_size, ls_A, ls_B = local_search(best_A, best_B)
            if ls_size > best_size:
                best_size = ls_size
                best_A = ls_A
                best_B = ls_B
    
    # Final validation
    if not check_sidon(best_A) or not check_sidon(best_B):
        return [[1], [2]]
    
    dA = get_diffs(best_A)
    dB = get_diffs(best_B)
    if dA & dB:
        return [[1], [2]]
    
    return [best_A, best_B]
