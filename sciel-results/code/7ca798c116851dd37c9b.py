import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    n = 256

    # Precompute conflict triples: (a, b, c) where a < b <= c and a | (b+c)
    # For each a, store the set of (b, c) pairs that conflict
    conflict_map = {}
    for a in range(1, n + 1):
        pairs = []
        for b in range(a + 1, n + 1):
            # c >= b, and a | (b+c) => c = k*a - b for some k
            # c >= b => k*a - b >= b => k*a >= 2b => k >= ceil(2b/a)
            # c <= n => k*a - b <= n => k <= (n+b)/a
            min_k = (2 * b + a - 1) // a  # ceil(2b/a)
            max_k = (n + b) // a
            for k in range(min_k, max_k + 1):
                c = k * a - b
                if b <= c <= n:
                    pairs.append((b, c))
        conflict_map[a] = pairs

    def count_conflicts(S):
        """Count the number of violated triples in S."""
        S_set = set(S)
        cnt = 0
        for a in S:
            if a not in S_set:
                continue
            for (b, c) in conflict_map[a]:
                if b in S_set and c in S_set:
                    cnt += 1
        return cnt

    def is_valid(S):
        """Check if S has zero conflicts."""
        return count_conflicts(S) == 0

    def can_add(a, S_set):
        """Check if adding a to S_set (as new smallest) creates no conflicts.
        Only need to check conflicts where a is the smallest element."""
        for (b, c) in conflict_map[a]:
            if b in S_set and c in S_set:
                return False
        return True

    def greedy_from_base(base_set, upper_limit):
        """Greedily add elements from upper_limit down to 1."""
        S = set(base_set)
        S_set = set(S)
        for a in range(upper_limit, 0, -1):
            if time.time() > deadline - 5:
                break
            if a not in S_set and can_add(a, S_set):
                S.add(a)
                S_set.add(a)
        return S

    def local_search(S, max_iters=500):
        """Hill-climbing with swaps to minimize conflicts, then maximize size."""
        S = set(S)
        best_size = len(S)
        best_conflicts = count_conflicts(S)
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            S_list = sorted(S)
            
            # Try removing one element and adding one or more others
            for remove_elem in S_list:
                if time.time() > deadline - 3:
                    break
                
                S_test = S - {remove_elem}
                # Try to add smaller elements greedily
                added = []
                for a in range(1, remove_elem + 1):
                    if a not in S_test and can_add(a, S_test):
                        S_test.add(a)
                        added.append(a)
                
                if len(S_test) > best_size or (len(S_test) == best_size and count_conflicts(S_test) < best_conflicts):
                    S = S_test
                    best_size = len(S)
                    best_conflicts = count_conflicts(S)
                    improved = True
                    break
            
            # Also try simple swaps: remove one, add another
            if not improved:
                for remove_elem in S_list:
                    if time.time() > deadline - 3:
                        break
                    S_test = S - {remove_elem}
                    for add_elem in range(1, n + 1):
                        if add_elem not in S_test and can_add(add_elem, S_test):
                            S_test.add(add_elem)
                            if len(S_test) > best_size:
                                S = S_test
                                best_size = len(S)
                                best_conflicts = count_conflicts(S)
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return S

    # Strategy: Start with safe upper tail, then greedily extend downward
    # Safe base: elements in (2n/3, n] = (170.67, 256] => 171 to 256
    base = set(range(171, n + 1))
    
    # Greedy phase 1: add from 170 down to 1
    S = greedy_from_base(base, 170)
    
    # Local search to try to improve
    S = local_search(S, max_iters=300)
    
    # Final greedy pass to fill any remaining gaps
    S_final = set(S)
    for a in range(1, n + 1):
        if time.time() > deadline - 2:
            break
        if a not in S_final and can_add(a, S_final):
            S_final.add(a)
    
    # Verify validity; if invalid, fall back to safe base
    if not is_valid(S_final):
        S_final = base
    
    return sorted(S_final)
