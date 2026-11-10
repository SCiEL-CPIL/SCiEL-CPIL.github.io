import time
import random
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    def is_valid_addition(S, x):
        """
        Check if adding x to S results in a valid set.
        S is a list of elements.
        """
        # Optimization: iterate over S
        for a in S:
            ga_x = gcds[a][x]
            for b in S:
                if a == b:
                    continue
                # Check if gcd(a,b) == gcd(a,x) == gcd(b,x)
                # We already have gcd(a,x). We need gcd(b,x) and gcd(a,b).
                # To avoid redundant gcd(b,x) calc, we can group by gcd(a,x)?
                # Or just compute.
                if gcds[a][b] == ga_x and gcds[b][x] == ga_x:
                    return False
        return True

    best_set = []
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    random.seed(42)
    
    # Run multiple iterations of Greedy + Local Search
    for run in range(200):
        if time.time() > deadline:
            break
            
        # 1. Greedy Initialization
        # Use a random order
        current_elements = elements[:]
        random.shuffle(current_elements)
        
        S = []
        S_set = set()
        for x in current_elements:
            if is_valid_addition(S, x):
                S.append(x)
                S_set.add(x)
        
        if len(S) > len(best_set):
            best_set = S[:]
            
        # 2. Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any element not in S
            # Iterate in random order to avoid bias
            candidates = [x for x in elements if x not in S_set]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if is_valid_addition(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    added = True
                    break # Restart loop to re-evaluate with new S
            
            if added:
                continue
            
            # Try 1-for-2 swap: Remove u, Add v, w
            # Iterate u in S
            # To speed up, pick random u? Or all u?
            # All u is better for finding improvements.
            
            swap_found = False
            # Shuffle S to randomize which u we try first
            S_shuffled = S[:]
            random.shuffle(S_shuffled)
            
            for u in S_shuffled:
                if time.time() > deadline:
                    break
                    
                # S_temp = S - {u}
                # Instead of creating new list, we can just check validity against S - {u}
                # is_valid_addition(S - {u}, v)
                
                # Try to add 2 elements
                # Candidates for v, w: elements not in S
                # We want to find v, w such that S - {u} + {v, w} is valid.
                # This means:
                # 1. v, w valid with S - {u}
                # 2. v, w valid with each other (in context of S - {u})
                
                # Greedy approach for adding 2:
                # Find v that is valid with S - {u}.
                # Then find w that is valid with (S - {u}) + {v}.
                
                # To make this faster, we can pre-filter candidates for v.
                # A candidate v is valid with S - {u} if is_valid_addition(S - {u}, v) is True.
                
                # Let's try to find a pair (v, w).
                # Iterate v in candidates
                # If valid with S - {u}:
                #    Iterate w in candidates (w != v)
                #    If valid with (S - {u}) + {v}:
                #       Found!
                
                # This is O(N^2 * CostCheck).
                # CostCheck is O(|S|^2).
                # N=128, |S|=80.
                # 128 * 128 * 6400 ~ 100M ops. Too slow for Python in a loop.
                
                # Optimization:
                # We can check validity of v with S - {u} faster?
                # is_valid_addition(S - {u}, v) checks pairs in S - {u}.
                
                # Let's just try a few random swaps instead of exhaustive.
                
                # Pick random v, w from candidates
                # If valid, do it.
                
                # Actually, let's just try to add 2 greedily.
                # 1. Find v valid with S - {u}.
                # 2. Find w valid with S - {u} + {v}.
                
                # To find v, we iterate candidates.
                # To find w, we iterate candidates.
                
                # Let's limit the search.
                
                S_temp_set = S_set - {u}
                S_temp_list = [x for x in S if x != u]
                
                # Find v
                v_found = None
                for v in candidates:
                    if is_valid_addition(S_temp_list, v):
                        v_found = v
                        break
                        
                if v_found is not None:
                    # Find w
                    # S_temp2 = S_temp + {v}
                    S_temp2_list = S_temp_list + [v_found]
                    S_temp2_set = S_temp_set | {v_found}
                    
                    w_found = None
                    for w in candidates:
                        if w == v_found:
                            continue
                        if is_valid_addition(S_temp2_list, w):
                            w_found = w
                            break
                            
                    if w_found is not None:
                        # Perform swap
                        S = S_temp2_list + [w_found]
                        S_set = S_temp2_set | {w_found}
                        swap_found = True
                        improved = True
                        break # Restart local search loop
                        
            if swap_found:
                continue
                
            # If no swap found, try 2-for-3? Or just break.
            # Let's try a random perturbation: remove 2 random, add 3?
            # Or just break to next run.
            break

    return best_set
