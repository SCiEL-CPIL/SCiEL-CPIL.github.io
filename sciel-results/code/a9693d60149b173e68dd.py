import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 88.0  # Safety margin
    n = 180
    
    # Precompute valid distinct triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  =>  a(b+c) = bc  =>  bc - ab - ac = 0  =>  (b-a)(c-a) = a^2
    # Let d = b-a, e = c-a. d*e = a^2.
    # b = a+d, c = a+e.
    # Distinctness: a != b => d != 0 (true since d>=1). a != c => e != 0. b != c => d != e.
    # So we need d != e.
    
    triples = []
    conflicts = [[] for _ in range(n)]
    
    for a in range(1, n + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d < e: # Ensures b < c and b != c
                    b = a + d
                    c = a + e
                    if b <= n and c <= n:
                        # Check distinctness explicitly just in case, though d<e implies b!=c
                        if a != b and a != c and b != c:
                            triples.append((a, b, c))
                            # Add to conflict lists
                            # For a: pair (b, c)
                            conflicts[a - 1].append((b - 1, c - 1))
                            # For b: pair (a, c)
                            conflicts[b - 1].append((a - 1, c - 1))
                            # For c: pair (a, b)
                            conflicts[c - 1].append((a - 1, b - 1))
            d += 1
            
    # Helper to check if adding x to S creates a violation
    # S is a list of booleans
    def can_add(S, x):
        # x is 0-indexed
        if S[x]:
            return True # Already there
        for (u, v) in conflicts[x]:
            if S[u] and S[v]:
                return False
        return True

    def add(S, x):
        S[x] = True

    def remove(S, x):
        S[x] = False

    def get_size(S):
        return sum(S)

    def greedy_descending():
        S = [False] * n
        for a in range(n - 1, -1, -1):
            if can_add(S, a):
                add(S, a)
        return S

    def local_search(S):
        cur_size = get_size(S)
        improved = True
        # Limit iterations to prevent infinite loops in rare cases
        max_iters = 1000 
        iters = 0
        
        while improved and iters < max_iters:
            iters += 1
            improved = False
            
            # Phase 1: Try to add any missing element
            # Randomize order to escape symmetries
            missing = [i for i in range(n) if not S[i]]
            random.shuffle(missing)
            for x in missing:
                if can_add(S, x):
                    add(S, x)
                    cur_size += 1
                    improved = True

            # Phase 2: Try to swap 1 out, 2 in (net gain +1)
            # This is crucial for growth when simple addition fails
            present = [i for i in range(n) if S[i]]
            random.shuffle(present)
            
            for x in present:
                # Try removing x
                remove(S, x)
                cur_size -= 1
                
                # Try adding as many as possible
                missing2 = [i for i in range(n) if not S[i]]
                random.shuffle(missing2)
                added = 0
                for y in missing2:
                    if can_add(S, y):
                        add(S, y)
                        added += 1
                        
                if added > 1:
                    # Net gain
                    cur_size += added
                    improved = True
                else:
                    # Restore x
                    add(S, x)
                    cur_size += 1
                    
        return cur_size

    # Strategy 1: Greedy Descending
    best_set = greedy_descending()
    best_size = get_size(best_set)
    
    # Strategy 2: Local Search on Greedy Result
    if time.time() - t0 < DEADLINE - 10:
        S_ls = best_set[:]
        size_ls = local_search(S_ls)
        if size_ls > best_size:
            best_size = size_ls
            best_set = S_ls[:]

    # Strategy 3: Multi-Start with Perturbation
    # If we have time, try restarting with slightly different greedy choices
    # We can simulate this by randomizing the "can_add" check order or 
    # by starting from a random valid set.
    
    # A better multi-start: 
    # 1. Generate a random valid set (e.g., random subset, then repair)
    # 2. Run local search
    
    # However, Greedy Descending is very strong.
    # Let's try "Greedy Ascending"?
    # If we add 1, 2, 3... 
    # When adding a, we check conflicts with b,c > a. But they are not in S yet.
    # So Ascending is trivial (no conflicts with future elements).
    # Wait, if we add 1, then 2, then 3...
    # When we add 3, we check if (1, 2) is in S? 
    # Conflict for 3: pairs (a,b) such that 1/3 = 1/a + 1/b.
    # a,b < 3? 
    # 1/3 = 1/2 + 1/6 -> 6>3.
    # 1/3 = 1/1 + 1/... no.
    # Generally, if a is small, b,c are larger.
    # So if we build Ascending, we don't know about b,c yet.
    # So Ascending Greedy would just take ALL numbers?
    # Let's check: If we take {1, ..., 180}, is it valid?
    # No, because (6,7,42) are all in it.
    # So Ascending Greedy fails immediately when it reaches 42, because 6 and 7 are already in.
    # So Ascending Greedy is:
    # for a in 1..180:
    #   if can_add(S, a): add
    # This is actually a valid strategy!
    # Let's try both and pick the best.

    S_asc = [False] * n
    for a in range(n):
        if can_add(S_asc, a):
            add(S_asc, a)
    size_asc = get_size(S_asc)
    
    if size_asc > best_size:
        best_size = size_asc
        best_set = S_asc[:]
        
    # Local search on Ascending result
    if time.time() - t0 < DEADLINE - 10:
        S_ls2 = best_set[:]
        size_ls2 = local_search(S_ls2)
        if size_ls2 > best_size:
            best_size = size_ls2
            best_set = S_ls2[:]

    # Final Polish
    if time.time() - t0 < DEADLINE - 2:
        S_final = best_set[:]
        size_final = local_search(S_final)
        if size_final > best_size:
            best_size = size_final
            best_set = S_final[:]

    out = [i + 1 for i in range(n) if best_set[i]]
    return out
