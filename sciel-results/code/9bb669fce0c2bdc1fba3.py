import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]
            
    # Precompute conflict structure
    # elem_conflicts[v] = list of (a, b) such that {v, a, b} is a bad triple
    # This allows O(deg(v)) check for can_add
    elem_conflicts = [[] for _ in range(N + 1)]
    conflict_degree = [0] * (N + 1)
    
    # Iterate over all triples to find bad ones
    # To speed up, we can just iterate all triples, N=128 is small enough for precomp
    # Number of triples is C(128,3) ~ 350k. This is fast.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    # Bad triple (a, b, c)
                    elem_conflicts[a].append((b, c))
                    elem_conflicts[b].append((a, c))
                    elem_conflicts[c].append((a, b))
                    conflict_degree[a] += 1
                    conflict_degree[b] += 1
                    conflict_degree[c] += 1
                    
    if time.time() > deadline:
        return [1]

    def can_add(v, selected_set):
        """
        Check if v can be added to selected_set.
        v is 1-indexed.
        selected_set is a set of 1-indexed integers.
        Uses precomputed elem_conflicts[v] which contains pairs (a, b)
        such that (v, a, b) is a bad triple.
        If both a and b are in selected_set, we cannot add v.
        """
        for a, b in elem_conflicts[v]:
            if a in selected_set and b in selected_set:
                return False
        return True

    def greedy(order, selected_set=None):
        if selected_set is None:
            selected_set = set()
        for x in order:
            if can_add(x, selected_set):
                selected_set.add(x)
        return selected_set

    def local_search(S):
        """
        Multi-stage local search:
        1. Add any possible element.
        2. 1-for-1 swap: remove x, add y if possible (size stays same, but might unlock better swaps? 
           Actually, standard local search for max independent set usually tries to increase size.
           If size is same, we might get stuck. However, Inspiration 1 did this. 
           Let's prioritize size-increasing moves first.
           If no size-increasing move, try 1-1 swap to change composition, then re-check adds?
           Or just try to increase size via 1-2 and 2-3 swaps.
           
           Let's stick to size-increasing moves primarily, but 1-1 swap can be useful to escape local minima 
           if combined with subsequent adds. However, to keep it simple and effective:
           - Try Add.
           - Try 1-for-2 swap (remove 1, add 2).
           - Try 2-for-3 swap (remove 2, add 3).
           - Try 1-for-1 swap (remove 1, add 1) ONLY if it leads to a subsequent add? 
             That's complex. Let's do 1-1 swap to just diversify if stuck, then try adds again.
        """
        S = set(S)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Add any possible
            for x in range(1, N + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if not improved:
                # 2. 1-for-2 swap: remove x, try to add two elements y, z
                found_swap = False
                for x in list(S):
                    S_trial = S - {x}
                    added = 0
                    # Try to add elements in some order. Random or fixed?
                    # Fixed order is deterministic and fast.
                    for y in range(1, N + 1):
                        if y not in S_trial and can_add(y, S_trial):
                            S_trial.add(y)
                            added += 1
                            if added >= 2:
                                break
                    if len(S_trial) > len(S):
                        S = S_trial
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    continue

            if not improved:
                # 3. 2-for-3 swap: remove x, y, try to add three elements
                found_swap = False
                s_list = list(S)
                # Sample pairs to avoid O(N^4) if S is large, but N=128 is small.
                # C(128, 2) ~ 8000. For each pair, try adds.
                # This might be slow if done exhaustively every iteration.
                # We will limit the number of pairs checked per iteration.
                for i in range(len(s_list)):
                    if time.time() > deadline - 2: break
                    for j in range(i + 1, len(s_list)):
                        S_trial = S - {s_list[i], s_list[j]}
                        added = 0
                        for y in range(1, N + 1):
                            if y not in S_trial and can_add(y, S_trial):
                                S_trial.add(y)
                                added += 1
                                if added >= 3:
                                    break
                        if len(S_trial) > len(S):
                            S = S_trial
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    continue

            if not improved:
                # 4. 1-for-1 swap: remove x, add y. 
                # This doesn't increase size, but might allow future adds.
                # We do this with a limit to avoid infinite loops or wasting time.
                # Only try a few random swaps if stuck.
                if len(S) > 0:
                    s_list = list(S)
                    random.shuffle(s_list)
                    for x in s_list[:10]: # Limit swaps tried
                        S_trial = S - {x}
                        for y in range(1, N + 1):
                            if y not in S_trial and can_add(y, S_trial):
                                S = S_trial
                                S.add(y)
                                improved = True
                                break
                        if improved:
                            break
                # If 1-1 swap was done, loop back to try Adds again

        return S

    best_set = set()
    best_size = 0
    
    # Run multiple randomized greedy passes
    num_restarts = 50
    random.seed(42)
    
    for i in range(num_restarts):
        if time.time() > deadline - 15: # Leave time for final local search on best
            break
            
        # Strategy 1: Random order
        if i % 3 == 0:
            order = list(range(1, N + 1))
            random.shuffle(order)
        # Strategy 2: Sort by conflict degree (low degree first is often good for independent set)
        elif i % 3 == 1:
            order = list(range(1, N + 1))
            order.sort(key=lambda x: conflict_degree[x] + random.random())
        # Strategy 3: Reverse degree
        else:
            order = list(range(1, N + 1))
            order.sort(key=lambda x: -conflict_degree[x] + random.random())
            
        S = greedy(order)
        
        # Apply local search
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Final polish on best_set
    if best_set:
        best_set = local_search(best_set)
    else:
        best_set = {1} # Fallback
        
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
