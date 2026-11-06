import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for safety

    def get_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) < 3:
            return True
        S_set = set(S)
        # O(n^2) check is fine for n<=240
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(vals, time_limit):
        n = len(vals)
        if n == 0:
            return []
        
        vals_arr = np.array(vals)
        vals_set = set(vals)
        
        # Precompute conflict graph? 
        # For N=240, N^2 = 57600 pairs.
        # We can precompute for each i, the set of j such that vals[i] + vals[j] is in vals.
        # But since we are doing dynamic swaps, a fast check function is better.
        
        # Vectorized conflict check for a single element x against a set S (represented as boolean mask or list)
        # Let's represent S as a boolean array of length n for fast membership.
        
        def get_conflicts(x_idx, S_mask):
            # Returns indices j in S such that vals[x_idx] + vals[j] is in vals
            # AND vals[j] + vals[x_idx] is in vals (symmetric)
            # AND 2*vals[x_idx] is in vals
            x_val = vals_arr[x_idx]
            
            # Check 2x
            if x_val * 2 in vals_set:
                # Find index of 2x
                # We can precompute a map from value to index
                pass
            
            # For efficiency, let's just use a set for S in the inner loop?
            # No, let's stick to a set of values for S, and check x+y in S.
            # The bottleneck in the current program is likely the Python overhead of set lookups and loops.
            # Let's try a hybrid: keep S as a set, but iterate over a smaller candidate list.

        # Better approach:
        # 1. Generate strong initial candidates.
        # 2. Use a "conflict degree" heuristic to pick elements to remove/add.
        
        # Precompute value to index map
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        # Precompute for each i, the list of j such that vals[i] + vals[j] is in vals
        # This is the "sums to" relation.
        # conflicts[i] = set of j such that vals[i] + vals[j] in vals
        # Note: if i==j, we check 2*vals[i] in vals.
        
        conflict_pairs = [set() for _ in range(n)]
        for i in range(n):
            vi = vals_arr[i]
            for j in range(n):
                vj = vals_arr[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # The condition is a+b=c. 
                    # If we pick i and j, we cannot pick k.
                    # So i conflicts with k? No, i and j cannot coexist with k.
                    # Actually, the constraint is: no a,b,c in S such that a+b=c.
                    # This is a 3-uniform hypergraph independent set problem.
                    # But it can be viewed as: if i and j are in S, then k is forbidden.
                    # Or: if i and k are in S, then j is forbidden (since i+j=k => j=k-i).
                    
                    # Let's define "conflict" as: pair (i, j) is invalid if vals[i]+vals[j] is in S.
                    # This depends on S.
                    
                    # Alternative view:
                    # S is sum-free iff for all i, j in S, vals[i]+vals[j] is NOT in S.
                    # This is equivalent to: S is an independent set in a graph where edges connect i and j 
                    # if vals[i] + vals[j] is in the *universe*? No.
                    
                    # Let's stick to the local search with fast validation.
                    # For N=240, checking if x can be added to S:
                    # 1. Check if 2x in S.
                    # 2. For each y in S, check if x+y in S.
                    # This is O(|S|) lookups.
                    # If |S| ~ 80, this is 80 lookups.
                    # 240 elements to try * 80 lookups = ~20k ops per full pass.
                    # This is very fast in Python if optimized.
                    
                    pass

        def can_add(x, S_set):
            if 2*x in S_set:
                return False
            for y in S_set:
                if x + y in S_set:
                    return False
            return True

        def local_search(S, time_limit):
            S = set(S)
            last_improvement = time.time()
            rng = random.Random(42)
            
            # Sort elements by "conflict degree" to prioritize removals?
            # Or just random.
            
            while time.time() < time_limit:
                improved = False
                
                # 1. Try to add any element
                # To speed up, we can maintain a list of "blocked" elements?
                # For now, just iterate.
                candidates = [x for x in vals if x not in S]
                rng.shuffle(candidates)
                
                # Optimization: If S is large, checking all candidates is expensive.
                # We can limit the number of candidates checked per pass.
                for x in candidates:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
                        last_improvement = time.time()
                        break  # Restart pass after addition? Or continue?
                        # Restarting is safer to ensure we don't miss interactions, but continuing might be faster.
                        # Let's continue to try to add more in the same pass.
                        # But if we add x, it might block y.
                        # Actually, adding x never "unblocks" anything. It only blocks more.
                        # So if we can add x, we should.
                        # But if we add x, we might prevent adding z later.
                        # Greedy addition is fine.
                
                if improved:
                    continue

                # 2. Try swaps
                # Remove y, see if we can add more than 1 element.
                # To find good y to remove, pick y that is involved in many conflicts.
                # Conflict count for y: number of pairs (a,b) in S such that a+b=y OR y+a=b OR y+b=a.
                
                # Calculate conflict scores for elements in S
                # This is O(|S|^2) per step, which is 80^2 = 6400. Acceptable.
                
                conflict_scores = {y: 0 for y in S}
                for y in S:
                    for z in S:
                        if y == z: continue
                        if y + z in S:
                            # y and z conflict with y+z
                            # This means y is "bad" because it forces us to exclude y+z
                            # Or y+z is bad because it forces us to exclude y,z
                            # Let's count how many pairs (a,b) in S sum to y? No.
                            # Let's count how many elements in S are "blocked" by y.
                            # If we remove y, which elements become addable?
                            # An element x is blocked by y if:
                            # 1. x+y in S
                            # 2. x-y in S (since y+x' = x => x' = x-y)
                            # 3. 2x in S (independent of y, but if x is in S, 2x is blocked)
                            
                            # This is getting complex. Let's just try random swaps.
                            pass

                # Simpler swap strategy:
                # Pick a random y in S.
                # Remove y.
                # Try to greedily add all possible elements.
                # If size increases, keep it.
                
                if S:
                    y = rng.choice(list(S))
                    S_test = S - {y}
                    
                    # Greedily add
                    S_temp = set(S_test)
                    added = 0
                    # We need to check all non-members
                    for x in vals:
                        if x not in S_temp:
                            if can_add(x, S_temp):
                                S_temp.add(x)
                                added += 1
                    
                    if added > 1: # Net gain > 0
                        S = S_temp
                        improved = True
                        last_improvement = time.time()
                        break

                # 3. If stuck, try a "kick": remove 2 elements, add 3?
                # Or random restart?
                # Let's try removing a random pair and greedily filling.
                if len(S) >= 2:
                    y1, y2 = rng.sample(list(S), 2)
                    S_test = S - {y1, y2}
                    S_temp = set(S_test)
                    added = 0
                    for x in vals:
                        if x not in S_temp:
                            if can_add(x, S_temp):
                                S_temp.add(x)
                                added += 1
                    if added > 2:
                        S = S_temp
                        improved = True
                        last_improvement = time.time()
                        break

            return S

        # Initial Candidates
        candidates = []
        
        # 1. Odds
        odds = [x for x in vals if x % 2 != 0]
        candidates.append(set(odds))
        
        # 2. Positives
        pos = [x for x in vals if x > 0]
        candidates.append(set(pos))
        
        # 3. Negatives
        neg = [x for x in vals if x < 0]
        candidates.append(set(neg))
        
        # 4. Top half by value
        sv = sorted(vals)
        candidates.append(set(sv[n//2:]))
        
        # 5. Bottom half by value
        candidates.append(set(sv[:n//2]))
        
        # 6. Top half by abs value
        sv_abs = sorted(vals, key=lambda x: abs(x), reverse=True)
        candidates.append(set(sv_abs[:n//2 + 1]))
        
        # 7. Bottom half by abs value
        candidates.append(set(sv_abs[n//2:]))
        
        # 8. Random Greedy
        for _ in range(10):
            rng = random.Random()
            order = vals[:]
            rng.shuffle(order)
            S = set()
            for x in order:
                if can_add(x, S):
                    S.add(x)
            candidates.append(S)
            
        # 9. Greedy by abs value
        for order in [sorted(vals, key=lambda x: abs(x), reverse=True), 
                      sorted(vals, key=lambda x: abs(x))]:
            S = set()
            for x in order:
                if can_add(x, S):
                    S.add(x)
            candidates.append(S)

        # Sort candidates by size
        candidates.sort(key=len, reverse=True)
        
        best_S = set()
        best_size = -1
        
        # Allocate time
        total_time = time_limit
        # Give more time to larger candidates
        weights = [len(S) for S in candidates]
        total_weight = sum(weights)
        
        for i, S in enumerate(candidates):
            if time.time() > deadline - 5.0:
                break
                
            # Validate
            if not is_sum_free(S):
                # Fix it
                S_fixed = set()
                for x in S:
                    if can_add(x, S_fixed):
                        S_fixed.add(x)
                S = S_fixed
            
            # Time allocation proportional to size, with a minimum
            t_alloc = max(1.0, total_time * (weights[i] / total_weight))
            t_end = min(time.time() + t_alloc, deadline - 5.0)
            
            S_improved = local_search(S, t_end)
            
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = S_improved
                
        return list(best_S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = get_instance(size, seed)
        
        if size == 60:
            t_lim = 20.0
        elif size == 120:
            t_lim = 40.0
        else:
            t_lim = 60.0
            
        S = solve_instance(vals, t_lim)
        
        # Final validation
        if not is_sum_free(S):
            # Fallback to odds
            S = [x for x in vals if x % 2 != 0]
            
        results.append(sorted(S))
        
        if time.time() > deadline:
            break

    while len(results) < 3:
        results.append([])

    return results[:3]
