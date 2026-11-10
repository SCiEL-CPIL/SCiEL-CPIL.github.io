import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 55.0 # 55s safety margin
    
    n = 128
    vertices = list(range(1, n + 1))
    
    # Precompute gcds for all pairs to speed up is_bad
    # gcd_table[i][j] = gcd(i, j)
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def is_bad_triple(a, b, c):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        if g_ab != g_ac:
            return False
        g_bc = gcd_table[b][c]
        return g_ab == g_bc

    def can_add(S, v):
        # S is a set of integers
        # Check if v forms a bad triple with any pair in S
        # To speed up, we can iterate pairs in S
        # If S is small, this is fine.
        S_list = list(S)
        len_S = len(S_list)
        for i in range(len_S):
            a = S_list[i]
            for j in range(i + 1, len_S):
                b = S_list[j]
                if is_bad_triple(a, b, v):
                    return False
        return True

    def greedy_init(order):
        S = set()
        for v in order:
            if can_add(S, v):
                S.add(v)
        return S

    def local_search(S):
        # Try to add elements
        # Then try swaps
        improved = True
        while improved and (time.time() - start_time) < limit:
            improved = False
            # Try to add
            candidates = [v for v in vertices if v not in S]
            random.shuffle(candidates)
            added = False
            for v in candidates:
                if can_add(S, v):
                    S.add(v)
                    improved = True
                    added = True
                    break
            if added:
                continue
                
            # Try to swap
            # Remove one, try to add two? Or just add one that was blocked?
            # If we remove u, we might allow adding v.
            # We want to increase size, so we need to add more than we remove.
            # So remove 1, add 2? Or remove 1, add 1 (no gain) but maybe enables future?
            # Let's try remove 1, add 1 first to escape local minima?
            # Actually, if we can't add anyone, we are at a local maximum for additions.
            # To increase size, we must remove k and add k+1.
            # Simplest: remove 1, try to add 2.
            
            # This is expensive.
            # Alternative: Random restart.
            # Or: Simulated annealing style.
            
            # Let's try a simple swap: remove u, add v.
            # This keeps size same. Might help if we allow it to change composition.
            # But we want MAX size.
            
            # Let's try: Remove u. Then try to add ANY v.
            # If we can add v, size is same.
            # If we can add v1 AND v2, size increases.
            
            # Efficient check:
            # For each u in S:
            #   S_temp = S - {u}
            #   Count how many v in candidates can be added to S_temp.
            #   If count >= 2, we can remove u and add 2.
            #   If count == 1, we can remove u and add 1 (neutral).
            
            # This is O(|S| * |V| * |S|^2) which might be slow.
            # |S| ~ 15, |V| ~ 128.
            # 15 * 128 * 100 = 192,000 ops. Fast enough.
            
            found_swap = False
            for u in list(S):
                S_temp = S - {u}
                # Try to add 2 elements
                added_count = 0
                new_elements = []
                # We need to check if v can be added to S_temp
                # And if v1, v2 can be added to S_temp simultaneously?
                # If can_add(S_temp, v1) and can_add(S_temp, v2) is not enough,
                # we need can_add(S_temp U {v1}, v2).
                
                # Let's just try to add one by one to S_temp.
                # This finds a valid set of additions.
                
                # To maximize chance of finding 2, shuffle candidates
                cands = [v for v in vertices if v not in S_temp and v != u]
                random.shuffle(cands)
                
                temp_S = set(S_temp)
                for v in cands:
                    if can_add(temp_S, v):
                        temp_S.add(v)
                        added_count += 1
                        new_elements.append(v)
                        if added_count == 2:
                            break
                
                if added_count >= 2:
                    # Remove u, add new_elements
                    S = temp_S
                    improved = True
                    found_swap = True
                    break
                elif added_count == 1:
                    # Neutral swap, maybe useful?
                    # Let's accept with small probability or if it leads to future gains?
                    # For now, skip neutral swaps to focus on growth.
                    pass
            
            if not found_swap:
                # Stuck. Random restart?
                # Or break and return best found.
                # Since we are in a loop, we might just break.
                # But we have time. Let's do a random restart.
                # Generate new greedy set with different order.
                order = vertices[:]
                random.shuffle(order)
                S = greedy_init(order)
                improved = True # Force loop to continue with new S

        return S

    best_S = set()
    
    # Run multiple restarts
    while time.time() - start_time < limit:
        order = vertices[:]
        random.shuffle(order)
        S = greedy_init(order)
        S = local_search(S)
        
        if len(S) > len(best_S):
            best_S = S
        
        # If we found a very large set, we might break early?
        # Theoretical max is likely around 15-20.
        # If we hit 20, we can stop.
        if len(best_S) >= 20:
            break

    return list(best_S)
