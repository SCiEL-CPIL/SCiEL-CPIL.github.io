import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    n = 192
    
    def is_valid(S):
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        S_arr = np.array(S_list)
        
        for i, d in enumerate(S_list):
            mask = np.ones(K, dtype=bool)
            mask[i] = False
            others = S_arr[mask]
            L = len(others)
            if L < 3:
                continue
            
            # Vectorized check: does d divide sum of any 3 distinct elements in others?
            residues = others % d
            # Count residues
            unique_res, counts = np.unique(residues, return_counts=True)
            res_count = dict(zip(unique_res, counts))
            
            # Check all pairs (u, v) in others
            for u_idx in range(L):
                u = others[u_idx]
                ru = u % d
                for v_idx in range(u_idx + 1, L):
                    v = others[v_idx]
                    rv = v % d
                    target_r = (- (ru + rv)) % d
                    
                    cnt = res_count.get(target_r, 0)
                    if cnt == 0:
                        continue
                    
                    if target_r != ru and target_r != rv:
                        return False
                    elif target_r == ru and target_r != rv:
                        if cnt >= 2:
                            return False
                    elif target_r != ru and target_r == rv:
                        if cnt >= 2:
                            return False
                    elif target_r == ru and target_r == rv:
                        if cnt >= 3:
                            return False
        
        return True

    def can_add(S, x):
        S_list = list(S)
        K = len(S_list)
        
        # Check 1: x divides sum of any 3 in S
        if K >= 3:
            # Direct O(K^3) check - fast for small K
            for i in range(K):
                for j in range(i+1, K):
                    for k in range(j+1, K):
                        if (S_list[i] + S_list[j] + S_list[k]) % x == 0:
                            return False
        
        # Check 2: For each y in S, y divides sum of any triple in S U {x} that includes x
        for y in S_list:
            A = [z for z in S_list if z != y]
            L = len(A)
            if L < 2:
                continue
            for i in range(L):
                u = A[i]
                for j in range(i+1, L):
                    v = A[j]
                    if (x + u + v) % y == 0:
                        return False
        
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, max_iters=5000, aggressive=False):
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        
        not_in_S = [x for x in range(1, n+1) if x not in S]
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit * 0.9:
                break
            
            improved = False
            
            # Try adding an element (greedy fill)
            if not_in_S:
                random.shuffle(not_in_S)
                for x in not_in_S:
                    if can_add(S, x):
                        S.add(x)
                        not_in_S.remove(x)
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        improved = True
                        break
                if improved:
                    no_improve_count = 0
                    continue
            
            # Try 1-swap: remove one, add one
            if not_in_S and len(S) > 0:
                S_list = list(S)
                random.shuffle(S_list)
                num_removals = 10 if aggressive else 8
                num_candidates = 20 if aggressive else 15
                for r in S_list[:num_removals]:
                    new_S = (S - {r})
                    candidates = [x for x in not_in_S if x != r]
                    if not candidates:
                        continue
                    random.shuffle(candidates)
                    for x in candidates[:num_candidates]:
                        if can_add(new_S, x):
                            new_S.add(x)
                            if is_valid(new_S):
                                S = new_S
                                not_in_S = [y for y in range(1, n+1) if y not in S]
                                if len(S) > best_size:
                                    best_S = set(S)
                                    best_size = len(S)
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                                improved = True
                                break
                    if improved:
                        break
            
            # Try 2-swap: remove two, add two
            if not improved and len(not_in_S) >= 2 and len(S) >= 2:
                S_list = list(S)
                random.shuffle(S_list)
                num_pairs = 6 if aggressive else 5
                num_cands1 = 15 if aggressive else 12
                num_cands2 = 15 if aggressive else 12
                for i in range(min(num_pairs, len(S_list))):
                    r1 = S_list[i]
                    for j in range(i+1, min(len(S_list), i+num_pairs+1)):
                        r2 = S_list[j]
                        new_S = (S - {r1, r2})
                        candidates = [x for x in not_in_S if x not in (r1, r2)]
                        if len(candidates) < 2:
                            continue
                        random.shuffle(candidates)
                        for x1 in candidates[:num_cands1]:
                            if can_add(new_S, x1):
                                new_S2 = new_S | {x1}
                                remaining_cands = [x for x in candidates if x != x1]
                                for x2 in remaining_cands[:num_cands2]:
                                    if can_add(new_S2, x2):
                                        new_S3 = new_S2 | {x2}
                                        if is_valid(new_S3):
                                            S = new_S3
                                            not_in_S = [y for y in range(1, n+1) if y not in S]
                                            if len(S) > best_size:
                                                best_S = set(S)
                                                best_size = len(S)
                                                no_improve_count = 0
                                            else:
                                                no_improve_count += 1
                                            improved = True
                                            break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try 3-swap: remove three, add three (only in aggressive mode)
            if not improved and aggressive and len(not_in_S) >= 3 and len(S) >= 3:
                S_list = list(S)
                random.shuffle(S_list)
                for i in range(min(4, len(S_list))):
                    r1 = S_list[i]
                    for j in range(i+1, min(len(S_list), i+5)):
                        r2 = S_list[j]
                        for k in range(j+1, min(len(S_list), j+5)):
                            r3 = S_list[k]
                            new_S = (S - {r1, r2, r3})
                            candidates = [x for x in not_in_S if x not in (r1, r2, r3)]
                            if len(candidates) < 3:
                                continue
                            random.shuffle(candidates)
                            found = False
                            for x1 in candidates[:10]:
                                if can_add(new_S, x1):
                                    new_S2 = new_S | {x1}
                                    rem1 = [x for x in candidates if x != x1]
                                    for x2 in rem1[:10]:
                                        if can_add(new_S2, x2):
                                            new_S3 = new_S2 | {x2}
                                            rem2 = [x for x in rem1 if x != x2]
                                            for x3 in rem2[:10]:
                                                if can_add(new_S3, x3):
                                                    new_S4 = new_S3 | {x3}
                                                    if is_valid(new_S4):
                                                        S = new_S4
                                                        not_in_S = [y for y in range(1, n+1) if y not in S]
                                                        if len(S) > best_size:
                                                            best_S = set(S)
                                                            best_size = len(S)
                                                        no_improve_count = 0
                                                        found = True
                                                        improved = True
                                                        break
                                            if found:
                                                break
                                        if found:
                                            break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > (30 if aggressive else 50):
                    break
        
        return best_S

    # Phase 1: Multiple greedy seeds with bias toward larger numbers
    best_S = set()
    best_size = 0
    
    # Greedy from high to low (largest numbers first - they have fewer divisors)
    S1 = greedy_construct(range(n, 0, -1))
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
    
    # Greedy from low to high
    S2 = greedy_construct(range(1, n+1))
    if len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
    
    # Biased greedy: prefer numbers in upper half
    upper_half = list(range(n//2 + 1, n+1))
    lower_half = list(range(1, n//2 + 1))
    order_biased = upper_half + lower_half
    S3 = greedy_construct(order_biased)
    if len(S3) > best_size:
        best_S = S3
        best_size = len(S3)
    
    # Randomized greedy - do more since startup is faster
    for _ in range(15):
        if time.time() - start_time > time_limit * 0.25:
            break
        # Mix of biased and uniform random
        if random.random() < 0.5:
            order = list(range(1, n+1))
            random.shuffle(order)
        else:
            # Biased: 70% chance to pick from upper half first
            upper = list(range(n//2 + 1, n+1))
            lower = list(range(1, n//2 + 1))
            random.shuffle(upper)
            random.shuffle(lower)
            order = upper + lower
        S_rand = greedy_construct(order)
        if len(S_rand) > best_size:
            best_S = S_rand
            best_size = len(S_rand)
    
    # Phase 2: Local search from best seed (aggressive)
    best_S = local_search(best_S, max_iters=3000, aggressive=True)
    
    # Phase 3: More local search iterations with restarts
    restart_count = 0
    while time.time() - start_time < time_limit * 0.88:
        # Try local search from current best (aggressive)
        improved_S = local_search(best_S, max_iters=1200, aggressive=True)
        if len(improved_S) > len(best_S):
            best_S = improved_S
            restart_count = 0
        else:
            restart_count += 1
        
        # Try a fresh random seed more frequently
        if random.random() < 0.3:
            if random.random() < 0.5:
                order = list(range(1, n+1))
                random.shuffle(order)
            else:
                upper = list(range(n//2 + 1, n+1))
                lower = list(range(1, n//2 + 1))
                random.shuffle(upper)
                random.shuffle(lower)
                order = upper + lower
            S_new = greedy_construct(order)
            S_new = local_search(S_new, max_iters=800, aggressive=True)
            if len(S_new) > len(best_S):
                best_S = S_new
                restart_count = 0
            else:
                restart_count += 1
        
        # If stuck, try a different strategy: remove some elements and re-add
        if restart_count > 4 and len(best_S) > 0:
            S_mod = set(best_S)
            S_list = list(S_mod)
            random.shuffle(S_list)
            # Remove 2-4 random elements
            num_remove = random.randint(2, min(4, len(S_list)))
            for _ in range(num_remove):
                if S_list:
                    S_mod.discard(S_list.pop())
            S_mod = local_search(S_mod, max_iters=600, aggressive=True)
            if len(S_mod) > len(best_S):
                best_S = S_mod
                restart_count = 0
    
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
