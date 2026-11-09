import time
import random

def construct():
    start_time = time.time()
    deadline = 80.0
    n = 128
    
    # Precompute all bad triples: (a, b, c) distinct, a<b<c, where
    # a+b, a+c, b+c are all in 1..n
    # For a set S, it's invalid if {a,b,c} ⊆ S AND {a+b, a+c, b+c} ⊆ S
    # We'll track for each triple the "sums" that need to be in S
    # Actually, let's think differently.
    
    # A set S is valid iff: for all distinct a,b,c in S, NOT(a+b in S AND a+c in S AND b+c in S)
    # 
    # For local search, we need fast validity checking.
    # Precompute: for each pair (a,b) with a<b, the sum a+b.
    # For a given set S, a triple (a,b,c) is "bad" if a+b∈S, a+c∈S, b+c∈S.
    
    # Strategy: represent S as a set. For checking if we can add x:
    #   Check all pairs (a,b) in S such that a+b=x, a+c=x, b+c=x for some c... 
    #   Actually: adding x creates a bad triple if there exist a,b in S (a≠b≠x) such that
    #   a+b, a+x, b+x are all in S∪{x}. Since x is new, we need:
    #   - a+x in S AND b+x in S AND a+b in S  (all three sums already in S)
    #   - a+b in S AND a+x = x (impossible) 
    #   - etc.
    #   So: adding x is bad iff ∃ distinct a,b ∈ S\{x} with a≠b such that
    #     (a+b) ∈ S AND (a+x) ∈ S AND (b+x) ∈ S
    #   Wait, also need to check if x itself is one of the "members" a,b,c.
    #   If x is one of a,b,c: say x=a. Then we need b,c ∈ S, and x+b, x+c, b+c all in S.
    #   So: adding x is bad iff:
    #     (i) ∃ distinct b,c ∈ S, b≠c, b,c≠x: (x+b)∈S AND (x+c)∈S AND (b+c)∈S
    #     (ii) ∃ distinct b,c ∈ S, b≠c, b,c≠x: (x+b)∈S AND (b+c)∈S AND (x+c)∈S  [same as (i)]
    #     Actually (i) covers all cases where x is one of the three members.
    #     And case where x is NOT one of the three: ∃ distinct a,b,c ∈ S, a,b,c≠x:
    #       (a+b)∈S AND (a+c)∈S AND (b+c)∈S — but this is already true/false regardless of x.
    #       So this doesn't depend on x. If the current set is valid, this can't happen.
    
    # So to check if adding x to a valid set S keeps it valid:
    #   Check: does ∃ distinct b,c ∈ S, b≠c, b,c≠x such that (x+b)∈S AND (x+c)∈S AND (b+c)∈S?
    #   Equivalently: for each pair (b,c) in S with b<c, b,c≠x, check if (b+c)∈S.
    #     If yes, then check if (x+b)∈S AND (x+c)∈S.
    
    # For removing x from S: always valid (removing elements can't create new bad triples).
    
    # For swap (remove x, add y): check if adding y to S\{x} is valid.
    
    # Let's use a different approach: precompute for each value v (1..128), 
    # the list of pairs (a,b) with a<b, a+b=v, a,b in 1..128.
    
    pairs_by_sum = [[] for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            s = a + b
            if s <= n:
                pairs_by_sum[s].append((a, b))
    
    def can_add(S, x):
        """Check if adding x to valid set S keeps it valid."""
        if x in S:
            return True
        # Check: ∃ distinct b,c ∈ S, b≠c, b,c≠x, such that (b+c)∈S AND (x+b)∈S AND (x+c)∈S
        # For efficiency: iterate over pairs (b,c) in S with b<c, b,c≠x
        # where (b+c) is in S. Then check (x+b) in S and (x+c) in S.
        
        # Alternative: for each b in S where b≠x, for each c in S where c>b, c≠x:
        #   if (b+c) in S and (x+b) in S and (x+c) in S: return False
        # This is O(|S|^2) per check. With |S|~70, that's ~2450 checks. Should be fast enough.
        
        S_list = list(S)
        for i in range(len(S_list)):
            b = S_list[i]
            if b == x:
                continue
            xb = x + b
            if xb > n or xb not in S:
                continue
            for j in range(i+1, len(S_list)):
                c = S_list[j]
                if c == x:
                    continue
                xc = x + c
                if xc > n or xc not in S:
                    continue
                bc = b + c
                if bc <= n and bc in S:
                    return False
        return True
    
    def is_valid(S):
        """Check if S is a valid set."""
        S_list = sorted(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                ab = a + b
                if ab > n or ab not in S:
                    continue
                for k in range(j+1, len(S_list)):
                    c = S_list[k]
                    ac = a + c
                    bc = b + c
                    if ac <= n and ac in S and bc <= n and bc in S:
                        return False
        return True
    
    def greedy_fill(S, remaining_candidates):
        """Greedily add elements from remaining_candidates to S."""
        added = 0
        for x in remaining_candidates:
            if x not in S and can_add(S, x):
                S.add(x)
                added += 1
        return added
    
    def local_search(S, max_iter=5000):
        """Local search with add, remove, and swap moves."""
        best_S = set(S)
        best_size = len(S)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            S_list = sorted(S)
            not_in_S = [x for x in range(1, n+1) if x not in S]
            
            improved = False
            
            # Try adding a random element
            if not_in_S:
                x = random.choice(not_in_S)
                if can_add(S, x):
                    S.add(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    improved = True
            
            # Try removing a random element (to enable future adds)
            if len(S) > 10:
                x = random.choice(S_list)
                S.remove(x)
                # Try to add some elements
                for y in not_in_S:
                    if y == x:
                        continue
                    if can_add(S, y):
                        S.add(y)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        break
                # If we didn't gain, put x back
                if len(S) <= best_size - 1:
                    S.add(x)
            
            # Try swap: remove x, add y
            if S_list and not_in_S:
                x = random.choice(S_list)
                S.remove(x)
                y = random.choice(not_in_S)
                if can_add(S, y):
                    S.add(y)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                else:
                    # Revert
                    S.remove(y) if y in S else None
                    S.add(x)
            
            # Occasionally try a larger perturbation
            if it % 100 == 99 and not_in_S and S_list:
                # Remove 2 random, try to add 3
                to_remove = random.sample(S_list, min(2, len(S_list)))
                for x in to_remove:
                    S.discard(x)
                added_any = False
                for y in not_in_S:
                    if y in S:
                        continue
                    if can_add(S, y):
                        S.add(y)
                        added_any = True
                if len(S) <= best_size - 2:
                    # Revert
                    for x in to_remove:
                        if x not in S:
                            S.add(x)
        
        return best_S
    
    # Multiple restarts
    best_overall = set()
    best_size_overall = 0
    
    for seed in range(200):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Random initialization
        S = set()
        if seed % 3 == 0:
            # Start with all odds
            S = set(range(1, n+1, 2))
        elif seed % 3 == 1:
            # Random subset
            S = set(random.sample(range(1, n+1), random.randint(30, 50)))
        else:
            # Start with all evens + some odds
            S = set(range(2, n+1, 2))
            S.update(random.sample(range(1, n+1, 2), random.randint(5, 15)))
        
        # Verify initial set is valid; if not, fix by removing offending elements
        S_list = sorted(S)
        # Simple fix: iteratively remove elements that are part of bad triples
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(S_list)):
                a = S_list[i]
                if a not in S:
                    continue
                for j in range(i+1, len(S_list)):
                    b = S_list[j]
                    if b not in S:
                        continue
                    ab = a + b
                    if ab > n or ab not in S:
                        continue
                    for k in range(j+1, len(S_list)):
                        c = S_list[k]
                        if c not in S:
                            continue
                        ac = a + c
                        bc = b + c
                        if ac <= n and ac in S and bc <= n and bc in S:
                            # Remove one of them
                            S.discard(c)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                S_list = sorted(S)
        
        # Greedy fill
        remaining = [x for x in range(1, n+1) if x not in S]
        random.shuffle(remaining)
        greedy_fill(S, remaining)
        
        # Local search
        S = local_search(S, max_iter=3000)
        
        if len(S) > best_size_overall:
            best_size_overall = len(S)
            best_overall = set(S)
        
        if time.time() > deadline:
            break
    
    # Final greedy fill on best
    remaining = [x for x in range(1, n+1) if x not in best_overall]
    random.shuffle(remaining)
    greedy_fill(best_overall, remaining)
    
    # Verify
    if not is_valid(best_overall):
        # Fallback to all odds
        best_overall = set(range(1, n+1, 2))
    
    result = list(best_overall)
    # Ensure Python ints
    result = [int(x) for x in result]
    return result
