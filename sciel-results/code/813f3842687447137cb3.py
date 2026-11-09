import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 128
    
    # Precompute for each pair (a,b) the set of c that would form a bad triple
    # Bad triple (a,b,c): a+b in S, a+c in S, b+c in S
    # For fixed a,b in S, c is bad if a+c in S and b+c in S (and a+b in S)
    
    def check_no_bad(S):
        """Verify S has no bad triple. Returns True if valid."""
        for a in S:
            for b in S:
                if a >= b:
                    continue
                if (a+b) not in S:
                    continue
                for c in S:
                    if c <= b:
                        continue
                    if (a+c) in S and (b+c) in S:
                        return False
        return True
    
    def greedy_build(order):
        S = set()
        for x in order:
            if x in S:
                continue
            # Check if adding x creates a bad triple
            ok = True
            for a in S:
                if a >= x:
                    continue
                for b in S:
                    if b <= a:
                        continue
                    if b >= x:
                        continue
                    # triple (a, b, x)
                    if (a+b) in S and (a+x) in S and (b+x) in S:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                S.add(x)
        return S
    
    def local_search_add(S):
        """Try to add elements to S."""
        S = set(S)
        improved = True
        while improved:
            if time.time() > DEADLINE:
                break
            improved = False
            for x in range(1, N+1):
                if x in S:
                    continue
                ok = True
                for a in S:
                    if a >= x:
                        continue
                    for b in S:
                        if b <= a:
                            continue
                        if b >= x:
                            continue
                        if (a+b) in S and (a+x) in S and (b+x) in S:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    S.add(x)
                    improved = True
        return S
    
    best_S = set()
    
    # Multiple restarts with different random orderings
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > DEADLINE:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        S = greedy_build(order)
        S = local_search_add(S)
        if len(S) > len(best_S):
            best_S = S
    
    # Final local search
    if time.time() < DEADLINE - 5:
        best_S = local_search_add(best_S)
    
    # Verify and fix if needed
    if not check_no_bad(best_S):
        # Remove elements until valid
        lst = sorted(best_S, reverse=True)
        for x in lst:
            if time.time() > DEADLINE:
                break
            if check_no_bad(best_S):
                break
            best_S.discard(x)
    
    return list(best_S)
