import random
import numpy as np

def construct():
    # Parameters
    N_range = 129 # Sums from 128 to 256 inclusive
    S_min = 128
    S_max = 256
    U = 128 # Universe [0, 128]
    
    # Target size K
    # We want to balance K and MaxMult.
    # Seed K=65, Max=33.
    # Let's try K=55, 60, 65.
    
    best_max_mult = float('inf')
    best_A = None
    
    # Helper to calculate multiplicities
    def get_multiplicities(A_set):
        # A_set is a set of ints
        # Returns dict S -> count
        counts = {s: 0 for s in range(S_min, S_max + 1)}
        A_list = list(A_set)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            for j in range(i, n):
                b = A_list[j]
                s = a + b
                if S_min <= s <= S_max:
                    counts[s] += 1
        return counts

    # Helper to check coverage
    def is_covered(counts):
        return all(c > 0 for c in counts.values())

    # Local Search
    def local_search(initial_A, steps=2000):
        A = set(initial_A)
        counts = get_multiplicities(A)
        if not is_covered(counts):
            return None, float('inf')
        
        current_max = max(counts.values())
        current_sq_sum = sum(c*c for c in counts.values())
        
        # Histogram of multiplicities to quickly find max
        # max_mult is bounded by len(A)
        max_possible = len(A)
        hist = [0] * (max_possible + 1)
        for c in counts.values():
            hist[c] += 1
        
        def update_hist(old_counts, new_counts, hist):
            # This is inefficient if we do it from scratch every time.
            # Better to update incrementally.
            pass

        # Incremental update strategy
        # We maintain `counts` and `hist`.
        
        for step in range(steps):
            if not A: break
            a = random.choice(list(A))
            candidates = [x for x in range(U + 1) if x not in A]
            if not candidates: break
            b = random.choice(candidates)
            
            # Calculate change in counts
            # Pairs involving a: {a, x} for x in A, x != a. And {a, a}.
            # Pairs involving b: {b, x} for x in A \ {a}, x != b. And {b, b}.
            # Note: When we remove a and add b, the set becomes A' = (A - {a}) + {b}.
            # Pairs removed:
            # 1. {a, a} -> sum 2a
            # 2. {a, x} for x in A, x != a -> sum a+x
            # Pairs added:
            # 1. {b, b} -> sum 2b
            # 2. {b, x} for x in A, x != a, x != b -> sum b+x
            # Note: {a, b} is not a pair in A nor A' if a!=b.
            # In A, pairs are subsets of A.
            # In A', pairs are subsets of A'.
            
            delta = {s: 0 for s in range(S_min, S_max + 1)}
            
            # Remove a
            # Pair {a, a}
            s_aa = 2 * a
            if S_min <= s_aa <= S_max:
                delta[s_aa] -= 1
                hist[delta[s_aa] + counts[s_aa]] -= 1 # Wait, hist update needs care.
                
            # Let's just recalculate the affected sums.
            # Affected sums are those involving a or b.
            # Sums involving a: 2a, a+x for x in A.
            # Sums involving b: 2b, b+x for x in A' \ {b} = A \ {a}.
            
            affected_sums = set()
            s_aa = 2 * a
            if S_min <= s_aa <= S_max: affected_sums.add(s_aa)
            for x in A:
                if x == a: continue
                s_ax = a + x
                if S_min <= s_ax <= S_max: affected_sums.add(s_ax)
                
            s_bb = 2 * b
            if S_min <= s_bb <= S_max: affected_sums.add(s_bb)
            for x in A:
                if x == a: continue
                s_bx = b + x
                if S_min <= s_bx <= S_max: affected_sums.add(s_bx)
                
            # Calculate new counts for affected sums
            new_counts = {}
            for s in affected_sums:
                # Count how many pairs in A sum to s
                old_c = counts[s]
                # Count how many pairs in A' sum to s
                # Pairs in A' summing to s:
                # 1. Pairs in A \ {a} that sum to s (and don't involve b, but b not in A yet)
                #    Actually, pairs in A' are formed from A'.
                #    A' = (A \ {a}) U {b}.
                #    Pairs in A' are:
                #    - Pairs from A \ {a}
                #    - Pairs {b, x} where x in A \ {a}
                #    - Pair {b, b}
                
                # Let's count directly for the small set of affected sums.
                # This is O(K) per sum. Affected sums is O(K). Total O(K^2) per step?
                # K ~ 60. K^2 ~ 3600. Steps 2000. Total 7M ops. Fast enough.
                
                c_new = 0
                A_prime_list = [x for x in A if x != a] + [b]
                # This creates a new list every time, which is slow.
                # Let's just iterate A.
                
                # Simpler:
                # old count = c_old
                # Change:
                # If s == 2a: -1
                # If s == a+x (x in A, x!=a): -1
                # If s == 2b: +1
                # If s == b+x (x in A, x!=a): +1
                
                change = 0
                if s == 2 * a:
                    change -= 1
                for x in A:
                    if x == a: continue
                    if s == a + x:
                        change -= 1
                    if s == b + x:
                        change += 1
                if s == 2 * b:
                    change += 1
                
                new_c = old_c + change
                new_counts[s] = new_c
                
            # Check if new state is valid (no negative counts, all covered)
            valid = True
            for s in affected_sums:
                if new_counts[s] < 0:
                    valid = False
                    break
                if new_counts[s] == 0:
                    # Check if it was 0 before? If it was >0 and becomes 0, coverage lost.
                    if counts[s] > 0:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Calculate new max and sq_sum
            # We need to update hist.
            # For each s in affected_sums:
            # hist[old_c] -= 1
            # hist[new_c] += 1
            
            temp_hist = hist[:] # Copy is expensive? List size ~ 100. OK.
            new_max = current_max
            new_sq_sum = current_sq_sum
            
            for s in affected_sums:
                old_c = counts[s]
                new_c = new_counts[s]
                
                new_sq_sum += new_c*new_c - old_c*old_c
                
                temp_hist[old_c] -= 1
                temp_hist[new_c] += 1
                
            # Find new max from temp_hist
            for m in range(len(temp_hist)-1, -1, -1):
                if temp_hist[m] > 0:
                    new_max = m
                    break
            
            # Acceptance criteria
            accept = False
            if new_max < current_max:
                accept = True
            elif new_max == current_max and new_sq_sum < current_sq_sum:
                accept = True
            elif new_max == current_max and new_sq_sum == current_sq_sum:
                # Random walk to explore
                if random.random() < 0.5:
                    accept = True
            elif new_max > current_max:
                # Allow slight increase if it reduces variance significantly?
                # Or just annealing.
                # Let's try to accept if new_sq_sum is much smaller, even if max increases slightly.
                # But we must not exceed best_max found so far globally?
                # No, local search can go up.
                # If new_max < best_max found in this run?
                # Let's keep it simple: Accept if new_max < current_max OR (new_max == current_max and better sq_sum).
                # This is greedy. Might get stuck.
                # Let's add a small probability of accepting worse max if sq_sum improves a lot.
                if new_sq_sum < current_sq_sum - 10 and new_max <= current_max + 2:
                    if random.random() < 0.1:
                        accept = True

            if accept:
                # Apply change
                for s in affected_sums:
                    counts[s] = new_counts[s]
                hist = temp_hist
                current_max = new_max
                current_sq_sum = new_sq_sum
                A.remove(a)
                A.add(b)
                
        return A, current_max

    # Main loop
    best_max = float('inf')
    best_A = None
    
    # Try a few random starts
    for trial in range(10):
        K = random.randint(55, 70)
        # Ensure we have elements that can form small and large sums
        # Random set is usually fine.
        A_init = set(random.sample(range(U + 1), K))
        
        # Check coverage quickly
        counts_init = get_multiplicities(A_init)
        if not is_covered(counts_init):
            # Try to fix or skip
            # If not covered, skip this start
            continue
            
        A_res, max_res = local_search(A_init, steps=3000)
        
        if A_res is not None and max_res < best_max:
            best_max = max_res
            best_A = A_res
            
        if best_max <= 10: # Good enough?
            break

    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
        
    return list(best_A)
