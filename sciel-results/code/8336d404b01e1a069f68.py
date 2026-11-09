import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    # Precompute 4-AP neighbors: for each x, list of 4-AP indices containing x
    four_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        for x in (a, b, c, d):
            four_ap_neighbors[x].append(idx)
    
    # Precompute 3-AP partners for fast delta calculation
    # For a 3-AP (a, b, c), the middle is b.
    # If we have a set S, a 3-AP exists if all three are in S.
    # To calculate delta for adding x:
    # We gain a 3-AP if there exists (x, y, z) or (y, x, z) or (y, z, x) in S.
    # Specifically:
    # 1. x is first term: need y, z in S such that (x, y, z) is 3-AP. y=x+d, z=x+2d.
    # 2. x is middle term: need a, c in S such that (a, x, c) is 3-AP. a=x-d, c=x+d.
    # 3. x is last term: need a, b in S such that (a, b, x) is 3-AP. a=x-2d, b=x-d.
    
    # Let's precompute for each x, the list of (partner1, partner2) for each role.
    # role 0: x is start. partners are (x+d, x+2d)
    # role 1: x is mid. partners are (x-d, x+d)
    # role 2: x is end. partners are (x-2d, x-d)
    
    # Actually, it's simpler to just precompute the "potential 3-APs" involving x.
    # For each x, let's store a list of tuples (p1, p2, role) where if p1 and p2 are in S, adding x completes a 3-AP.
    # But for delta, we need to know how many NEW 3-APs are formed.
    # A 3-AP is (a, b, c). It is "completed" by adding x if x is one of a,b,c and the other two are in S.
    # So for each x, we can precompute a list of (u, v) pairs such that (x, u, v) is a 3-AP (in any order).
    # Then delta_add(x) = count of (u, v) in this list such that u in S and v in S.
    
    three_ap_partners = {i: [] for i in range(1, N + 1)}
    for a, b, c in three_ap_tuples:
        # (a, b, c) is a 3-AP.
        # If we add a, we need b and c in S.
        three_ap_partners[a].append((b, c))
        # If we add b, we need a and c in S.
        three_ap_partners[b].append((a, c))
        # If we add c, we need a and b in S.
        three_ap_partners[c].append((a, b))
    
    # Helper: check if adding x to S creates a 4-AP
    def creates_4ap(S, x):
        for idx in four_ap_neighbors[x]:
            a, b, c, d = four_ap_tuples[idx]
            # Check if the other 3 are in S
            # We know x is one of them.
            if x == a:
                if b in S and c in S and d in S: return True
            elif x == b:
                if a in S and c in S and d in S: return True
            elif x == c:
                if a in S and b in S and d in S: return True
            elif x == d:
                if a in S and b in S and c in S: return True
        return False

    # Helper: count 3-APs in S (for initial calculation and verification)
    def count_3aps(S):
        count = 0
        for a, b, c in three_ap_tuples:
            if a in S and b in S and c in S:
                count += 1
        return count

    # Helper: calculate delta for adding x to S
    def delta_add(S, x):
        delta = 0
        for u, v in three_ap_partners[x]:
            if u in S and v in S:
                delta += 1
        return delta

    # Helper: calculate delta for removing x from S
    def delta_remove(S, x):
        # We lose all 3-APs containing x that are currently in S.
        # This is exactly delta_add(S - {x}, x)
        # But we can calculate it directly: count (u, v) in three_ap_partners[x] such that u in S and v in S.
        # Wait, delta_add(S, x) counts how many 3-APs are formed by adding x to S.
        # These are exactly the 3-APs that exist in S union {x} and contain x.
        # So delta_remove(S, x) is the number of 3-APs in S that contain x.
        # This is the same calculation as delta_add(S - {x}, x).
        # Let's just use the same logic: count partners in S.
        delta = 0
        for u, v in three_ap_partners[x]:
            if u in S and v in S:
                delta += 1
        return delta

    # Strategy: Multi-start Greedy Local Search
    # 1. Generate several diverse initial sets using different greedy heuristics.
    # 2. For each, perform a fast local search using incremental deltas.
    # 3. Keep the best.
    
    best_set = set()
    best_score = -1
    
    # --- Initial Set 1: Standard Greedy (from seed) ---
    S1 = set()
    for x in range(1, N + 1):
        if not creates_4ap(S1, x):
            S1.add(x)
    
    # --- Initial Set 2: Reverse Greedy ---
    S2 = set()
    for x in range(N, 0, -1):
        if not creates_4ap(S2, x):
            S2.add(x)
    
    # --- Initial Set 3: Randomized Greedy ---
    S3 = set()
    perm = list(range(1, N + 1))
    random.shuffle(perm)
    for x in perm:
        if not creates_4ap(S3, x):
            S3.add(x)
    
    # --- Initial Set 4: "Density" Greedy - prioritize elements with high 3-AP potential ---
    # Calculate static 3-AP degree for each x (how many 3-APs it's part of in the full set)
    # This is just len(three_ap_partners[x])
    # Let's sort by this degree descending, and greedily add if no 4-AP.
    S4 = set()
    sorted_by_3ap_deg = sorted(range(1, N + 1), key=lambda x: len(three_ap_partners[x]), reverse=True)
    for x in sorted_by_3ap_deg:
        if not creates_4ap(S4, x):
            S4.add(x)
    
    initial_sets = [S1, S2, S3, S4]
    
    # Local Search Function
    def local_search(S):
        S = set(S)
        score = count_3aps(S)
        no_improve_count = 0
        max_no_improve = 5000 # Stop if stuck for a while
        
        while time.time() < deadline - 1.0 and no_improve_count < max_no_improve:
            improved = False
            
            # 1. Try all possible additions
            # We want to add x if it doesn't create a 4-AP and has positive delta.
            # To be efficient, we can iterate through candidates.
            # Candidates are all x not in S.
            # We can sort candidates by delta descending to find the best addition quickly.
            # But calculating delta for all candidates is O(N * deg(x)) which is ~ 256 * 100 = 25600 ops.
            # This is fast enough.
            
            best_add = None
            best_add_delta = 0
            best_remove = None
            best_remove_delta = 0 # We want to remove if delta is small (low cost) or if it enables a better add?
            # Actually, simple hill climbing:
            # Step 1: Find best addition.
            # Step 2: If no good addition, find best removal (lowest cost) to unblock? 
            # Or just swap?
            
            # Let's stick to:
            # 1. Find best addition x (max delta, no 4-AP).
            # 2. If found, add it.
            # 3. Else, find best swap (remove x, add y) where delta(y) - delta(x) > 0 and no 4-AP.
            
            # Calculate deltas for all candidates
            candidates_add = []
            for x in range(1, N + 1):
                if x not in S:
                    if not creates_4ap(S, x):
                        d = delta_add(S, x)
                        if d > 0:
                            candidates_add.append((d, x))
            
            if candidates_add:
                # Pick the one with highest delta
                candidates_add.sort(reverse=True)
                d, x = candidates_add[0]
                S.add(x)
                score += d
                improved = True
                no_improve_count = 0
                continue
            
            # No beneficial addition. Try swaps.
            # For each x in S, calculate cost to remove (delta_remove).
            # For each y not in S, calculate gain to add (delta_add).
            # We want gain(y) - cost(x) > 0.
            # This is O(N^2) if done naively.
            # Optimization:
            # Precompute gain(y) for all y not in S.
            # Precompute cost(x) for all x in S.
            # Then find max gain(y) - min cost(x) subject to 4-AP constraint of S - {x} + {y}.
            # The 4-AP constraint is the hard part.
            
            # Let's just try a limited number of random swaps if stuck.
            # Or, try removing the element with the lowest "value" (cost) and see if we can add something.
            
            # Simpler: Random local search step.
            # Pick random x in S, random y not in S.
            # If S-{x}+{y} is valid and score improves, accept.
            
            if len(S) < 2:
                break
                
            x = random.choice(list(S))
            y = random.choice([i for i in range(1, N + 1) if i not in S])
            
            # Check validity of S - {x} + {y}
            # 1. Does S - {x} have a 4-AP? No, S was valid.
            # 2. Does adding y to S - {x} create a 4-AP?
            #    We need to check if y completes a 4-AP with elements in S - {x}.
            #    This is similar to creates_4ap, but we must exclude x from the check.
            
            valid = True
            for idx in four_ap_neighbors[y]:
                a, b, c, d = four_ap_tuples[idx]
                # Check if the other 3 are in S - {x}
                others = []
                for val in (a, b, c, d):
                    if val != y:
                        others.append(val)
                
                # If x is one of the others, it's not in S-{x}, so this 4-AP is not formed by y.
                # If x is y, impossible.
                # If none of the others is x, then they must be in S.
                # If one of the others is x, then we only need the other 2 to be in S.
                
                count_in_S_minus_x = 0
                for val in others:
                    if val == x:
                        continue # Not in S - {x}
                    if val in S:
                        count_in_S_minus_x += 1
                    else:
                        break # If a required element is not in S, it's not in S-{x} either.
                
                if count_in_S_minus_x == len(others) - (1 if x in others else 0):
                    # All required elements are in S - {x}
                    # Wait, if x is in others, we need the other 2 to be in S.
                    # If x is not in others, we need all 3 to be in S.
                    # My loop logic:
                    # If val == x, skip.
                    # If val in S, count++.
                    # If val not in S, break.
                    # This works if we assume S is the base.
                    # Let's re-verify.
                    # We want to know if (others \ {x}) subset of S.
                    # If yes, then y completes the 4-AP in S-{x}.
                    
                    # The condition `count_in_S_minus_x == len(others) - (1 if x in others else 0)`
                    # is equivalent to: all elements in others (except x) are in S.
                    # Because if any was not in S, we would have broken early.
                    # So yes, if we reach the end without breaking, it's valid.
                    valid = False
                    break
            
            if valid:
                # Calculate new score
                # New Score = Score - delta_remove(S, x) + delta_add(S - {x}, y)
                # Note: delta_add(S - {x}, y) is not the same as delta_add(S, y) if x is a partner of y.
                # So we must calculate it carefully.
                
                # Let's just calculate the delta directly to be safe, as it's O(deg(y))
                # Delta = (3-APs in S-{x}+{y}) - (3-APs in S)
                # = (3-APs in S-{x}+{y} that contain y) + (3-APs in S-{x} that don't contain y) - (3-APs in S)
                # This is getting complex.
                # Given the speed of the set operations, let's just do a quick check.
                # Actually, let's stick to the "Add only" hill climbing for the main loop,
                # and use swaps only if we get stuck.
                # The "Add only" part is very fast.
                # If we get stuck, we can try a random restart or a random swap.
                
                # For now, let's just accept a random swap if it's valid, even if score doesn't change?
                # No, we want to maximize score.
                # Let's calculate the exact delta for the swap.
                
                # Remove x
                S_temp = set(S)
                S_temp.remove(x)
                
                # Add y
                S_temp.add(y)
                
                new_score = count_3aps(S_temp)
                
                if new_score > score:
                    S = S_temp
                    score = new_score
                    improved = True
                    no_improve_count = 0
                elif new_score == score:
                    # Accept with low probability to escape local optima?
                    # Or just reject.
                    pass
                else:
                    pass
                    
            if not improved:
                no_improve_count += 1
                # If stuck for a long time, break out of this local search
                if no_improve_count > 200:
                    break
                    
        return S, score

    # Run local search on each initial set
    for i, S_init in enumerate(initial_sets):
        if time.time() > deadline - 5.0:
            break
            
        S_opt, score_opt = local_search(S_init)
        
        if score_opt > best_score:
            best_score = score_opt
            best_set = set(S_opt)
            
    return sorted(best_set)
