import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    N = 256
    # 1. Precompute conflict matrix
    # C[i][j] = True if i+j divides i*j
    # We only need i < j
    conflicts = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            s = i + j
            p = i * j
            if p % s == 0:
                conflicts[i][j] = True
                conflicts[j][i] = True
                
    # 2. Initialize set with all odds
    S = set()
    for i in range(1, N + 1, 2):
        S.add(i)
        
    # 3. Greedy add: try to add evens that have 0 conflicts with current S
    # We prioritize powers of 2 as they are known to be safe, but let's just check all evens
    # Actually, checking all numbers not in S is better.
    # We'll iterate a few times to allow cascading additions? 
    # If we add e1, it might block e2. If we add e2, it might block e1.
    # Since we start with odds, and evens are independent of each other? 
    # No, evens can conflict with each other.
    # But powers of 2 don't conflict with odds. Do they conflict with each other?
    # 2, 4: 6 | 8? No.
    # 2, 8: 10 | 16? No.
    # 4, 8: 12 | 32? No.
    # 2, 16: 18 | 32? No.
    # It seems powers of 2 are mutually non-conflicting.
    
    # Let's just do a general greedy pass
    added = True
    while added:
        added = False
        for v in range(1, N + 1):
            if v in S:
                continue
            # Check if v has 0 conflicts in S
            ok = True
            for u in S:
                if conflicts[v][u]:
                    ok = False
                    break
            if ok:
                S.add(v)
                added = True
                
    # 4. Local Search
    # Try to find moves that increase size.
    # Move type: Remove u from S, Add v to S. (Size change 0, but might enable future moves)
    # Move type: Remove u from S, Add v, w to S. (Size change +1)
    
    # Helper to check if v can be added to S (excluding u if u is being removed)
    def can_add(v, S, exclude=None):
        for u in S:
            if u == exclude:
                continue
            if conflicts[v][u]:
                return False
        return True

    # We will run a simple hill climbing
    # Since N is small, we can be somewhat aggressive with checks.
    
    while time.time() < deadline:
        improved = False
        
        # Try 1-for-2 moves
        # For each u in S, see if removing it allows adding two new nodes
        # This is expensive: O(|S| * (N-|S|)^2 * |S|)
        # |S| ~ 140, N-|S| ~ 116.
        # 140 * 116^2 * 140 ~ 25 million operations per pass. Too slow in Python.
        
        # Alternative: 
        # 1. Identify "bottlenecks". 
        # 2. Or just try random swaps/adds.
        
        # Let's try a simpler heuristic:
        # For each v not in S, let N_S(v) be neighbors in S.
        # If len(N_S(v)) == 1, say {u}, then removing u allows adding v.
        # This is a 1-for-1 swap.
        # If we do a 1-for-1 swap, size stays same. 
        # But it might open up a 1-for-2 move later.
        
        # Let's try to find a 1-for-2 move directly.
        # For each u in S:
        #   Let Cands = [v for v not in S if not can_add(v, S, u) and can_add(v, S)]
        #   Wait, can_add(v, S) is False for all v not in S (since we did greedy add).
        #   So all v not in S have at least one conflict in S.
        #   If we remove u, v becomes addable if u was the ONLY conflict.
        #   So Cands_u = [v not in S | N_S(v) == {u}]
        #   If we remove u, we can add any subset of Cands_u that is independent.
        #   We want to add at least 2 from Cands_u.
        
        # Optimization: Precompute for each v not in S, its neighbors in S.
        # This changes as S changes.
        
        # Given time constraints, let's do a randomized local search with 1-for-1 and 1-for-2.
        
        # Randomize order of u in S
        S_list = list(S)
        random.shuffle(S_list)
        
        for u in S_list:
            if time.time() > deadline:
                break
            
            # Find candidates v that are ONLY blocked by u
            # i.e., N_S(v) == {u}
            cands = []
            for v in range(1, N + 1):
                if v in S:
                    continue
                # Check if u is the only conflict
                only_u = True
                for w in S:
                    if w == u:
                        continue
                    if conflicts[v][w]:
                        only_u = False
                        break
                if only_u:
                    cands.append(v)
            
            if len(cands) >= 2:
                # We can remove u and add at least 2 from cands
                # We need to pick 2 from cands that don't conflict with each other
                # And don't conflict with S \ {u} (guaranteed by "only_u" property? 
                # No, "only_u" means no conflict with S\{u}. 
                # But cands might conflict with each other.)
                
                # Find max independent set in cands? 
                # Just try pairs
                found = False
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        v1, v2 = cands[i], cands[j]
                        if not conflicts[v1][v2]:
                            # Valid move
                            S.remove(u)
                            S.add(v1)
                            S.add(v2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break # Restart loop with new S
            
            # If no 1-for-2, try 1-for-1 to shuffle
            # Only if we have time
            if not improved and time.time() < deadline - 5:
                 # Try a random 1-for-1 swap to escape local minima?
                 # Pick a random v with N_S(v) == {u}
                 if cands:
                     v = random.choice(cands)
                     # Swap u with v
                     # This keeps size same.
                     # Is it useful? Maybe.
                     # But let's prioritize 1-for-2.
                     pass

    return list(S)

if __name__ == "__main__":
    print(construct())
