import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_size = 64  # Upper half: 65..128
    best_set = set(range(65, 129))
    
    def is_valid(s):
        """Check if set s has no three distinct a,b,c with all pair sums also in s."""
        sl = sorted(s)
        sset = set(sl)
        n = len(sl)
        for i in range(n):
            a = sl[i]
            for j in range(i + 1, n):
                b = sl[j]
                if a + b not in sset:
                    continue
                for k in range(j + 1, n):
                    c = sl[k]
                    if a + c in sset and b + c in sset:
                        return False
        return True
    
    def greedy_with_propagation(deadline, seed):
        """
        Greedy construction with constraint propagation.
        Start with upper half, then try to add elements from 1..64.
        When considering adding x, check if it creates a violation with any pair in current set.
        """
        random.seed(seed)
        current = set(range(65, 129))
        
        # Precompute for each potential new element x, which pairs (a,b) in current
        # would be violated if x is added.
        # A violation occurs if there exist a,b in current such that:
        #   a+b in current, and x+a in current, and x+b in current
        # OR if x itself is a, b, or c in some triple.
        
        # Actually, the condition is: no three DISTINCT members a,b,c such that
        # a+b, a+c, b+c are ALL in S.
        # So when adding x to S, we need to check all triples that include x.
        # Triples including x: (x, a, b) where a,b in current, a!=b, a!=x, b!=x
        #   Need: x+a in S, x+b in S, a+b in S (all after adding x)
        # Also triples (a, b, c) not including x - already valid by induction.
        
        # So for adding x, check: for all pairs (a,b) in current with a<b:
        #   if a+b in current (or a+b == x, but x not in current yet, so a+b in current)
        #   and x+a in current (or x+a == x, impossible since a>0)
        #   and x+b in current (or x+b == x, impossible since b>0)
        #   then violation.
        
        # Wait, x+a could equal some element in current, or x+b could equal some element.
        # Since we're adding x, the new set is current ∪ {x}.
        # So x+a is in new set iff x+a in current or x+a == x (impossible).
        # Similarly x+b in new set iff x+b in current.
        # a+b in new set iff a+b in current or a+b == x.
        
        # So violation when adding x: exists a,b in current, a<b, a!=b:
        #   (a+b in current or a+b == x) AND (x+a in current) AND (x+b in current)
        
        # Precompute current set for fast lookup
        cur = set(current)
        
        candidates = list(range(1, 65))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in current:
                continue
            
            # Check if adding x creates a violation
            valid = True
            # Check all pairs (a,b) in current
            cur_list = sorted(current)
            n = len(cur_list)
            for i in range(n):
                a = cur_list[i]
                for j in range(i + 1, n):
                    b = cur_list[j]
                    # Check if (a,b,x) or (x,a,b) etc. form a violating triple
                    # The triple must have three distinct elements.
                    # Since x not in current, x is distinct from a and b.
                    # a and b are distinct (i != j).
                    # So {x, a, b} is a triple of distinct elements.
                    # Check if a+b, a+x, b+x are all in current ∪ {x}
                    # a+b in current ∪ {x} iff a+b in current or a+b == x
                    # a+x in current ∪ {x} iff a+x in current (since a+x != x for a>0)
                    # b+x in current ∪ {x} iff b+x in current (since b+x != x for b>0)
                    
                    if (a + b in cur or a + b == x) and (a + x in cur) and (b + x in cur):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current.add(x)
        
        return current
    
    def local_search(s, deadline, seed):
        """Local search: try to add elements, and try 1-for-2 swaps."""
        random.seed(seed)
        current = set(s)
        cur_size = len(current)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            cur = set(current)
            
            # Try adding each element not in current
            for x in range(1, 129):
                if time.time() > deadline:
                    break
                if x in current:
                    continue
                # Check if adding x is valid
                valid = True
                cur_list = sorted(current)
                n = len(cur_list)
                for i in range(n):
                    a = cur_list[i]
                    for j in range(i + 1, n):
                        b = cur_list[j]
                        if (a + b in cur or a + b == x) and (a + x in cur) and (b + x in cur):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current.add(x)
                    improved = True
                    if len(current) > cur_size:
                        cur_size = len(current)
                        cur = set(current)
            
            # Try 1-for-2 swaps: remove one, add two
            if time.time() > deadline:
                break
            cur_list = sorted(current)
            for rem in cur_list:
                if time.time() > deadline:
                    break
                trial = set(current)
                trial.remove(rem)
                added = 0
                for add in range(1, 129):
                    if time.time() > deadline:
                        break
                    if add == rem or add in trial:
                        continue
                    # Check if adding add to trial is valid
                    valid = True
                    trial_set = set(trial)
                    trial_list = sorted(trial)
                    n = len(trial_list)
                    for i in range(n):
                        a = trial_list[i]
                        for j in range(i + 1, n):
                            b = trial_list[j]
                            if (a + b in trial_set or a + b == add) and (a + add in trial_set) and (b + add in trial_set):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        trial.add(add)
                        added += 1
                if len(trial) > len(current):
                    current = trial
                    improved = True
                    if len(current) > cur_size:
                        cur_size = len(current)
        
        return current
    
    # Run multiple strategies
    strategies = [
        ("greedy_seed0", lambda d: greedy_with_propagation(d, 0)),
        ("greedy_seed1", lambda d: greedy_with_propagation(d, 1)),
        ("greedy_seed2", lambda d: greedy_with_propagation(d, 2)),
        ("greedy_seed3", lambda d: greedy_with_propagation(d, 3)),
        ("greedy_seed4", lambda d: greedy_with_propagation(d, 4)),
    ]
    
    for name, func in strategies:
        if time.time() > deadline - 10:
            break
        result = func(deadline - 5)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # Local search from best
    if time.time() < deadline - 5:
        result = local_search(best_set, deadline - 2, 42)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # More local search with different seeds
    if time.time() < deadline - 5:
        result = local_search(best_set, deadline - 2, 43)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # Final validation
    if not is_valid(best_set):
        best_set = set(range(65, 129))
        best_size = 64
    
    return sorted(best_set)
