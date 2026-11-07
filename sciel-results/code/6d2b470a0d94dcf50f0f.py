"""
Large sets in 1..128 without a bad triple.

A "bad triple" is a set {a,b,c} of three distinct members such that a+b, a+c,
and b+c are also in the set. We want to maximize the set size.

Strategy:
  1. Precompute for every triple (i,j,k) whether it is "bad" given the current
     set (this depends on which of the six values i, j, k, i+j, i+k, j+k are
     in the set).
  2. Start with a good initial set (the upper half, extended a bit).
  3. Use local search:
     - Additions: try to add an element that doesn't create a bad triple.
     - Swaps: try replacing an element with another to maintain/increase size.
     - Removals: if stuck, remove an element that participates in the most
       potential bad triples.
  4. Use a deadline guard to return the best valid set found.
"""

import time
import random

def construct():
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    N = 128
    all_vals = list(range(1, N+1))
    
    # Precompute: for each pair (i,j) with i<j, the three sums
    # A triple (i,j,k) is bad iff i,j,k in S AND i+j in S AND i+k in S AND j+k in S
    # We'll track for each element x, which pairs would make a bad triple if x were added.
    
    def is_valid(S_set, S_list):
        """Check if S_set (a set of ints) has no bad triple."""
        for idx in range(len(S_list)):
            a = S_list[idx]
            for jdx in range(idx+1, len(S_list)):
                b = S_list[jdx]
                # a+b must not be in S_set (if a+b > N it's fine)
                if a + b > N:
                    continue
                if (a+b) not in S_set:
                    continue
                for kdx in range(jdx+1, len(S_list)):
                    c = S_list[kdx]
                    if a + c > N or b + c > N:
                        continue
                    if (a+c) in S_set and (b+c) in S_set:
                        return False
        return True
    
    def count_bad_triples(S_set):
        """Count the number of bad triples in S_set."""
        count = 0
        for idx in range(len(S_list_cache)):
            a = S_list_cache[idx]
            for jdx in range(idx+1, len(S_list_cache)):
                b = S_list_cache[jdx]
                if a + b > N:
                    continue
                if (a+b) not in S_set:
                    continue
                for kdx in range(jdx+1, len(S_list_cache)):
                    c = S_list_cache[kdx]
                    if a + c > N or b + c > N:
                        continue
                    if (a+c) in S_set and (b+c) in S_set:
                        count += 1
        return count
    
    # Global cache for S_list (we'll update it as we go)
    global S_list_cache
    S_list_cache = []
    
    # --- Initial set: try several seeds ---
    # Seed 1: {32..64} U {97..128} = size 65
    # Seed 2: upper half {65..128} = size 64
    # Seed 3: random-ish greedy
    
    def greedy_build(start_set):
        """Greedily add elements that don't create bad triples."""
        S_set = set(start_set)
        S_list = sorted(start_set)
        S_list_cache = list(S_list)
        
        candidates = [x for x in range(1, N+1) if x not in S_set]
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if adding x creates a bad triple
            creates_bad = False
            for i in range(len(S_list)):
                a = S_list[i]
                if a + x > N:
                    continue
                if (a+x) not in S_set:
                    continue
                for j in range(i+1, len(S_list)):
                    b = S_list[j]
                    if a + b > N:
                        continue
                    if (a+b) not in S_set:
                        continue
                    if (b+x) > N:
                        continue
                    if (b+x) in S_set:
                        creates_bad = True
                        break
                if creates_bad:
                    break
            if not creates_bad:
                S_set.add(x)
                S_list.append(x)
                S_list.sort()
                S_list_cache = list(S_list)
        return S_list, S_set
    
    def local_search(S_list, S_set, deadline):
        """Improve the set via local search."""
        S_list_cache = list(S_list)
        best_size = len(S_list)
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline:
            if no_improve_count >= max_no_improve:
                break
            
            S_list_cache = list(S_list)
            
            # --- Try additions ---
            added = False
            candidates = [x for x in range(1, N+1) if x not in S_set]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() >= deadline:
                    break
                # Check if adding x is valid
                valid = True
                for i in range(len(S_list)):
                    a = S_list[i]
                    if a + x > N:
                        continue
                    if (a+x) not in S_set:
                        continue
                    for j in range(i+1, len(S_list)):
                        b = S_list[j]
                        if a + b > N:
                            continue
                        if (a+b) not in S_set:
                            continue
                        if b + x > N:
                            continue
                        if (b+x) in S_set:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    S_list.append(x)
                    S_list.sort()
                    S_set.add(x)
                    added = True
                    no_improve_count = 0
                    best_size = max(best_size, len(S_list))
                    break
            
            if added:
                continue
            
            # --- Try swaps: remove one, add one ---
            swapped = False
            S_list_copy = list(S_list)
            random.shuffle(S_list_copy)
            for remove_x in S_list_copy:
                if time.time() >= deadline:
                    break
                S_set_trial = S_set - {remove_x}
                S_list_trial = [y for y in S_list if y != remove_x]
                # Try to add some candidate
                candidates = [x for x in range(1, N+1) if x not in S_set_trial]
                random.shuffle(candidates)
                for add_x in candidates:
                    if len(S_list_trial) + 1 <= best_size - 1:
                        continue  # No improvement
                    valid = True
                    for i in range(len(S_list_trial)):
                        a = S_list_trial[i]
                        if a + add_x > N:
                            continue
                        if (a+add_x) not in S_set_trial:
                            continue
                        for j in range(i+1, len(S_list_trial)):
                            b = S_list_trial[j]
                            if a + b > N:
                                continue
                            if (a+b) not in S_set_trial:
                                continue
                            if b + add_x > N:
                                continue
                            if (b+add_x) in S_set_trial:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        S_list = S_list_trial + [add_x]
                        S_list.sort()
                        S_set = S_set_trial | {add_x}
                        swapped = True
                        no_improve_count = 0
                        best_size = max(best_size, len(S_list))
                        break
                if swapped:
                    break
            
            if swapped:
                continue
            
            # --- Try removal to unlock future additions ---
            # Remove element that is in the most "potential" bad triples
            # For each element, count how many pairs (a,b) have a+b, a+x, b+x all in S
            removal_scores = {}
            for x in S_list:
                score = 0
                for i in range(len(S_list)):
                    a = S_list[i]
                    if a == x:
                        continue
                    if a + x > N:
                        continue
                    if (a+x) not in S_set:
                        continue
                    for j in range(i+1, len(S_list)):
                        b = S_list[j]
                        if b == x:
                            continue
                        if a + b > N:
                            continue
                        if (a+b) not in S_set:
                            continue
                        if b + x > N:
                            continue
                        if (b+x) in S_set:
                            score += 1
                removal_scores[x] = score
            
            if removal_scores:
                max_score = max(removal_scores.values())
                if max_score > 0:
                    # Remove a random element with max score
                    top_elements = [x for x, s in removal_scores.items() if s == max_score]
                    remove_x = random.choice(top_elements)
                    S_list.remove(remove_x)
                    S_set.discard(remove_x)
                    no_improve_count += 1
                else:
                    # No element causes issues; stuck
                    no_improve_count += 100
            else:
                no_improve_count += 100
        
        return S_list, S_set
    
    # --- Run multiple restarts with different seeds ---
    best_set = []
    best_size = 0
    
    seeds = [
        list(range(32, 65)) + list(range(97, 129)),  # size 65
        list(range(65, 129)),                         # size 64
        list(range(1, 129)),                          # full, will need heavy pruning
    ]
    
    for seed in seeds:
        if time.time() >= deadline:
            break
        
        # For the full seed, we need to prune first
        if len(seed) > N:
            seed = seed[:N]
        
        S_list, S_set = greedy_build(seed)
        S_list, S_set = local_search(S_list, S_set, deadline)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = list(S_list)
        
        # If we already have a very large set, we can stop
        if best_size >= 80 and time.time() < deadline:
            # Try a few more restarts
            pass
    
    # Final restart with random seed if time allows
    if time.time() < deadline - 5:
        random.seed()
        for _ in range(3):
            if time.time() >= deadline - 5:
                break
            # Random initial set: pick random subset
            seed = random.sample(all_vals, random.randint(40, 60))
            S_list, S_set = greedy_build(seed)
            S_list, S_set = local_search(S_list, S_set, deadline - 5)
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_set = list(S_list)
    
    # Verify the final set is valid
    if best_set:
        best_set.sort()
        S_final = set(best_set)
        # Quick validity check
        valid = True
        for i in range(len(best_set)):
            a = best_set[i]
            for j in range(i+1, len(best_set)):
                b = best_set[j]
                if a + b > N:
                    continue
                if (a+b) not in S_final:
                    continue
                for k in range(j+1, len(best_set)):
                    c = best_set[k]
                    if a + c > N or b + c > N:
                        continue
                    if (a+c) in S_final and (b+c) in S_final:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            # Fallback to a known valid set
            best_set = list(range(32, 65)) + list(range(97, 129))
    
    return [int(x) for x in best_set]
