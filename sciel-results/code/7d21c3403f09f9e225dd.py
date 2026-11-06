import random
import math
import time

def construct():
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute GCD matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            G[i][j] = g
            G[j][i] = g

    # Precompute bad triples? 
    # A triple (a,b,c) is bad if G[a][b] == G[a][c] == G[b][c].
    # There are C(128,3) ~ 340k triples. Storing them is fine.
    # But checking penalty requires iterating over triples in S.
    # Faster: For each pair (a,b) in S, let g = G[a][b].
    # Any c in S such that G[a][c]==g and G[b][c]==g forms a bad triple.
    # We can count bad triples efficiently during evaluation.

    def count_bad_triples(S_set, S_list):
        # S_list is list of elements in S_set
        # To be fast, we can precompute for each (a,b) the list of c's that form a bad triple with them?
        # No, it depends on S.
        # Iterative check:
        # For each a in S:
        #   For each b in S, b>a:
        #     g = G[a][b]
        #     Count c in S, c>b, such that G[a][c]==g and G[b][c]==g.
        # This is O(|S|^3). If |S| ~ 50, 125k ops. 5000 iters -> 600M ops. Too slow in Python.
        
        # Optimization:
        # Precompute for each pair (a,b) the set of c's that would make a bad triple.
        # bad_with[a][b] = set of c such that G[a][b]==G[a][c]==G[b][c]
        # Then penalty(S) = sum over pairs (a,b) in S of |bad_with[a][b] intersection S| / 3? 
        # Each bad triple is counted 3 times (once for each pair).
        # So Penalty = (Sum_{a,b in S} |bad_with[a][b] \cap S|) / 3.
        
        pass

    # Let's precompute bad_with
    bad_with = [[set() for _ in range(N+1)] for _ in range(N+1)]
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g = G[a][b]
            for c in range(b+1, N+1):
                if G[a][c] == g and G[b][c] == g:
                    bad_with[a][b].add(c)
                    bad_with[b][a].add(c)
                    bad_with[a][c].add(b) # Wait, the triple is unordered. 
                    # Actually, for the pair (a,c), b is the third element.
                    # If we iterate pairs, we just need to check if the third element is in S.
                    # Let's stick to: For pair (a,b), check if any c in S is in bad_with[a][b].
                    # Note: bad_with[a][b] should contain ALL c (c!=a,b) such that triple is bad.
                    # My loop above only added c > b. I should add all c != a,b.
    
    # Re-do precomputation correctly
    bad_with = [[set() for _ in range(N+1)] for _ in range(N+1)]
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g = G[a][b]
            for c in range(1, N+1):
                if c == a or c == b: continue
                if G[a][c] == g and G[b][c] == g:
                    bad_with[a][b].add(c)
                    bad_with[b][a].add(c)
    
    def get_penalty(S_set):
        # S_set is a set of integers
        # Iterate over pairs in S_set
        # To avoid O(N^2) per evaluation, we can maintain the penalty incrementally?
        # In SA, we only change one element at a time.
        # Adding x:
        #   New bad triples involving x: Pairs (a,b) in S such that x in bad_with[a][b].
        #   OR Pairs (a,x) in S_new such that some b in S is in bad_with[a][x].
        #   Actually, simpler:
        #   Penalty increases by number of pairs (a,b) in S such that x in bad_with[a][b].
        #   Wait, if (a,b,x) is a bad triple, it is counted when we look at pair (a,b) and see x.
        #   It is also counted when we look at pair (a,x) and see b.
        #   It is also counted when we look at pair (b,x) and see a.
        #   So if we define Penalty as sum over pairs (a,b) in S of |bad_with[a][b] intersection S|,
        #   then each bad triple contributes 3 to the sum.
        #   When adding x to S:
        #   1. New pairs (a,x) for a in S. For each a, check if any b in S is in bad_with[a][x].
        #      This finds triples (a,x,b).
        #   2. Existing pairs (a,b) in S. Check if x is in bad_with[a][b].
        #      This finds triples (a,b,x).
        #   These are the same sets of triples.
        #   So, Delta_Penalty = (Number of a in S such that bad_with[a][x] intersects S) 
        #                        + (Number of pairs (a,b) in S such that x in bad_with[a][b])
        #   Wait, if (a,b,x) is bad, then x in bad_with[a][b] AND b in bad_with[a][x] AND a in bad_with[b][x].
        #   So the first term counts each bad triple once (for the pair a,x).
        #   The second term counts each bad triple once (for the pair a,b).
        #   Total Delta = 2 * (Number of bad triples formed by x and S).
        #   Wait, let's verify.
        #   Term 1: Sum_{a in S} |bad_with[a][x] \cap S|. 
        #   If (a,b,x) is bad, then b in bad_with[a][x]. So this term counts 1 for this triple.
        #   Term 2: Sum_{a<b in S} 1 if x in bad_with[a][b].
        #   If (a,b,x) is bad, then x in bad_with[a][b]. So this term counts 1 for this triple.
        #   Total Delta = 2 * K, where K is number of bad triples involving x.
        #   This allows O(|S| * (|S| + |bad_with|)) update?
        #   Term 1 is O(|S| * |S|) if we iterate S for intersection.
        #   Term 2 is O(|S|^2).
        #   Since |S| < 128, this is fast.

        # We will maintain current_penalty.
        pass

    # Initialize S
    # Start with a valid set, e.g., powers of 2
    S = set([1, 2, 4, 8, 16, 32, 64, 128])
    # Verify validity
    # 1,2,4 -> 1,1,2 (OK)
    # 2,4,8 -> 2,2,4 (OK)
    # All powers of 2 are safe.
    
    current_penalty = 0
    # Verify initial penalty is 0
    # We can just run the check once.
    
    def calculate_penalty_from_scratch(S_set):
        cnt = 0
        lst = list(S_set)
        for i in range(len(lst)):
            a = lst[i]
            for j in range(i+1, len(lst)):
                b = lst[j]
                # Check if any c in S_set is in bad_with[a][b]
                # Optimization: intersect bad_with[a][b] with S_set
                # Since S_set is a set, intersection is fast.
                # But we need the count.
                cnt += len(bad_with[a][b].intersection(S_set))
        return cnt // 3 # Each triple counted 3 times

    current_penalty = calculate_penalty_from_scratch(S)
    assert current_penalty == 0, "Initial set should be valid"

    best_S = set(S)
    best_size = len(S)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    rng = random.Random(42)
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < 50:
        # Generate move
        # 1. Add random x not in S
        # 2. Remove random x in S
        # 3. Swap
        
        action = rng.random()
        if action < 0.4:
            # Add
            candidates = [x for x in nums if x not in S]
            if not candidates: continue
            x = rng.choice(candidates)
            # Calculate Delta Penalty
            # Term 1: Pairs (a,x) where a in S. Bad if some b in S is in bad_with[a][x].
            # Term 2: Pairs (a,b) in S. Bad if x in bad_with[a][b].
            
            delta_pen = 0
            # Term 1
            for a in S:
                # Check if bad_with[a][x] intersects S
                # To be faster, we can iterate over bad_with[a][x] and check if in S
                # Or iterate S and check if in bad_with[a][x]
                # S is small.
                for b in S:
                    if b == x: continue
                    if b in bad_with[a][x]:
                        delta_pen += 1
            # Term 2
            for a in S:
                for b in S:
                    if a >= b: continue
                    if x in bad_with[a][b]:
                        delta_pen += 1
            
            # Delta Size = +1
            # Objective: Maximize Size - Lambda * Penalty
            # Let's use Score = Size - 100 * Penalty
            old_score = len(S) - 100 * current_penalty
            new_score = (len(S) + 1) - 100 * (current_penalty + delta_pen)
            delta_score = new_score - old_score
            
            if delta_score >= 0 or rng.random() < math.exp(delta_score / T):
                S.add(x)
                current_penalty += delta_pen
                if current_penalty == 0 and len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        
        elif action < 0.8:
            # Remove
            if not S: continue
            x = rng.choice(list(S))
            # Calculate Delta Penalty
            # Removing x removes all bad triples involving x.
            # Delta Penalty is negative.
            delta_pen = 0
            # Term 1 (inverse): Pairs (a,x) in S. Bad if b in S in bad_with[a][x].
            for a in S:
                if a == x: continue
                for b in S:
                    if b == x: continue
                    if b in bad_with[a][x]:
                        delta_pen += 1
            # Term 2 (inverse): Pairs (a,b) in S. Bad if x in bad_with[a][b].
            for a in S:
                if a == x: continue
                for b in S:
                    if b == x: continue
                    if a >= b: continue
                    if x in bad_with[a][b]:
                        delta_pen += 1
            
            # Note: The logic above counts each bad triple twice (once for each direction of the pair involving x? No.)
            # Let's re-verify the delta logic.
            # When adding x, we added 2*K to penalty.
            # So removing x should subtract 2*K.
            # The loops above calculate 2*K.
            
            old_score = len(S) - 100 * current_penalty
            new_score = (len(S) - 1) - 100 * (current_penalty - delta_pen)
            delta_score = new_score - old_score
            
            if delta_score >= 0 or rng.random() < math.exp(delta_score / T):
                S.remove(x)
                current_penalty -= delta_pen
                if current_penalty == 0 and len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        
        else:
            # Swap
            if not S: continue
            x = rng.choice(list(S))
            candidates = [y for y in nums if y not in S]
            if not candidates: continue
            y = rng.choice(candidates)
            
            # Calculate Delta for removing x and adding y
            # This is complex. Easier to just do remove then add?
            # But we need atomic swap for SA.
            # Let's just do Remove then Add in sequence, but accept/reject based on combined delta?
            # For simplicity in this implementation, we might skip Swap or implement it as two steps.
            # Given time, let's stick to Add/Remove. Swap is less critical if Add/Remove is efficient.
            pass
        
        T *= cooling
        
    # Final cleanup: ensure best_S is valid
    if calculate_penalty_from_scratch(best_S) != 0:
        # Repair
        lst = list(best_S)
        changed = True
        while changed:
            changed = False
            for i in range(len(lst)):
                a = lst[i]
                for j in range(i+1, len(lst)):
                    b = lst[j]
                    for k in range(j+1, len(lst)):
                        c = lst[k]
                        if G[a][b] == G[a][c] == G[b][c]:
                            lst.remove(c)
                            changed = True
                            break
                    if changed: break
                if changed: break
        best_S = set(lst)
        
    if not best_S:
        best_S = {2}
        
    return list(best_S)
