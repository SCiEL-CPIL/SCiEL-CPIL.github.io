"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
import random

def construct():
    start_time = time.time()
    deadline = 58.0  # safety margin before 90s kill
    
    n_max = 180
    
    # Precompute forbidden triples {a, b, c} with a < b < c and 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    forbidden_triples = []
    triple_count = [0] * (n_max + 1)
    
    for a in range(1, n_max + 1):
        a_sq = a * a
        # Iterate u up to sqrt(a_sq)
        for u in range(1, int(a_sq**0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u >= v:
                    continue
                b = a + u
                c = a + v
                if b <= n_max and c <= n_max:
                    # b < c is guaranteed by u < v
                    # Distinctness: a < b, a < c. b != c.
                    forbidden_triples.append((a, b, c))
                    triple_count[a] += 1
                    triple_count[b] += 1
                    triple_count[c] += 1

    # Build conflict mapping: num -> list of (x, y) such that {x, y, num} is a forbidden triple
    conflicts = [[] for _ in range(n_max + 1)]
    for a, b, c in forbidden_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    def is_valid_addition(S, num):
        for x, y in conflicts[num]:
            if x in S and y in S:
                return False
        return True

    def run_greedy(order):
        S = set()
        for num in order:
            if time.time() - start_time > deadline:
                break
            if is_valid_addition(S, num):
                S.add(num)
        return S

    def local_search(S, max_iter=500):
        """Aggressive local search: try 1-add, 1-swap, 2-remove+2-add moves."""
        S = set(S)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() - start_time > deadline:
                break
            improved = False
            iter_count += 1
            
            # Try 1-addition
            for num in range(1, n_max + 1):
                if num not in S and is_valid_addition(S, num):
                    S.add(num)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try 1-swap: remove one, add one
            for remove in list(S):
                if time.time() - start_time > deadline:
                    break
                S.remove(remove)
                for add in range(1, n_max + 1):
                    if add not in S and is_valid_addition(S, add):
                        S.add(add)
                        improved = True
                        break
                if not improved:
                    S.add(remove)
            if improved:
                continue
                
            # Try 2-remove + 2-add
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() - start_time > deadline:
                    break
                for j in range(i + 1, len(S_list)):
                    r1, r2 = S_list[i], S_list[j]
                    S.remove(r1)
                    S.remove(r2)
                    found = False
                    for a in range(1, n_max + 1):
                        if found:
                            break
                        if a in S:
                            continue
                        if not is_valid_addition(S, a):
                            continue
                        S.add(a)
                        for b in range(1, n_max + 1):
                            if b in S:
                                continue
                            if not is_valid_addition(S, b):
                                continue
                            S.add(b)
                            found = True
                            break
                        if not found:
                            S.discard(a)
                    if not found:
                        S.add(r1)
                        S.add(r2)
                    else:
                        improved = True
                        break
                if improved:
                    break
        return S

    # Generate different orders for greedy
    order_deg = list(range(1, n_max + 1))
    order_deg.sort(key=lambda x: triple_count[x])
    
    order_desc = list(range(n_max, 0, -1))
    order_asc = list(range(1, n_max + 1))
    
    best_set = set()
    best_size = 0
    
    # Run greedy with deterministic orders + local search
    for order in [order_deg, order_desc, order_asc]:
        S = run_greedy(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Run greedy with random orders + local search
    for _ in range(30):
        if time.time() - start_time > deadline:
            break
        order_rand = list(range(1, n_max + 1))
        random.shuffle(order_rand)
        S = run_greedy(order_rand)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Final local search on best set with more iterations
    best_set = local_search(best_set, max_iter=1000)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
