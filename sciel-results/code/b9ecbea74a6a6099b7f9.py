import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs
    # ap_list: list of tuples (a, b, c, d)
    # conflicts: dict mapping element -> list of APs containing it
    ap_list = []
    conflicts = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            c = a + 2 * d
            b = a + d
            term4 = a + 3 * d
            if term4 > n:
                break
            ap = (a, b, c, term4)
            ap_list.append(ap)
            for term in ap:
                conflicts[term].append(ap)
                
    def can_add(x, S):
        """Check if x can be added to S without forming a 4-AP."""
        for ap in conflicts[x]:
            # Check if the other 3 elements are in S
            # We can use a set for S for O(1) lookup
            # But S is passed as a set.
            # Optimization: 
            # If any other element is not in S, this AP is not complete.
            # We only fail if ALL other 3 are in S.
            count_in_S = 0
            for term in ap:
                if term != x and term in S:
                    count_in_S += 1
            if count_in_S == 3:
                return False
        return True

    best_set = set()
    best_size = 0
    
    # Multi-start local search
    while time.time() < deadline:
        # 1. Greedy Construction
        S = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S):
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # 2. Local Search (Swaps)
        # Try to improve S by swapping elements
        # We run this for a fixed number of steps or until time runs out
        # To avoid long loops on a single restart, limit steps per restart
        steps_in_local_search = 0
        max_local_steps = 5000 # Arbitrary limit per restart to allow multiple restarts
        
        while steps_in_local_search < max_local_steps and time.time() < deadline:
            steps_in_local_search += 1
            
            # Pick a random element x not in S
            # We can maintain a list of candidates outside S, or just pick random 1..n
            # Picking random 1..n is simpler. If x in S, skip.
            # To be more efficient, we can keep a list of non-selected elements.
            # But maintaining it is overhead. For N=512, random pick is fine.
            
            # Find a candidate x not in S
            # To avoid infinite loop if S is very large, we can just try.
            # If S is size 300, probability x not in S is 212/512 ~ 0.4.
            
            # Better: Maintain a list of available elements?
            # Let's just pick random x.
            x = random.randint(1, n)
            if x in S:
                continue
                
            # Check if x can be added directly
            if can_add(x, S):
                S.add(x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = set(S)
                # Don't break, keep searching for more additions?
                # If we add x, S changes. We should continue.
                # But we incremented steps.
                continue
                
            # If x cannot be added, find a conflicting AP
            # and try to swap
            conflicting_ap = None
            for ap in conflicts[x]:
                others = [t for t in ap if t != x]
                if all(t in S for t in others):
                    conflicting_ap = ap
                    break
                    
            if not conflicting_ap:
                # Should not happen if can_add returned False
                continue
                
            # Pick a random element y from conflicting_ap to remove
            # y must be in S. It is, by definition of conflicting_ap.
            # We should not pick x (x is not in S).
            y = random.choice([t for t in conflicting_ap if t != x])
            
            # Try to swap: remove y, add x
            # Check if S \ {y} U {x} is valid.
            # We know that removing y breaks the specific conflicting_ap.
            # But x might form a NEW AP with S \ {y}.
            # So we need to check can_add(x, S \ {y}).
            
            # To do this efficiently:
            # We can temporarily remove y from S, check can_add(x, S), then restore y.
            # Or we can write a specialized check.
            # Given N=512, temporary modification is fast enough.
            
            S.discard(y)
            if can_add(x, S):
                S.add(x)
                # Size is unchanged (removed y, added x).
                # But the configuration changed.
                # We don't update best_size here since size is same.
                # However, this new configuration might allow future additions.
            else:
                # Swap failed, restore y
                S.add(y)
                
    return sorted(best_set)
