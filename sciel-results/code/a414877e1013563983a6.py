"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 90s kill
    
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

    def local_search_1swap(S, max_iters=5000):
        """Try 1-swap moves: remove one element, add another."""
        S = S.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() - start_time > deadline:
                break
            improved = False
            iters += 1
            # Try removing each element in S and adding each element not in S
            for rem in list(S):
                S.remove(rem)
                for add in range(1, n_max + 1):
                    if add not in S and is_valid_addition(S, add):
                        S.add(add)
                        improved = True
                        break
                if improved:
                    break
                else:
                    S.add(rem)
        return S

    def local_search_2swap(S, max_iters=2000):
        """Try 2-swap moves: remove two elements, add two others (net zero or gain)."""
        S = S.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() - start_time > deadline:
                break
            improved = False
            iters += 1
            S_list = sorted(S)
            # Try removing one element and adding one (1-swap within 2-swap framework)
            for i in range(len(S_list)):
                rem = S_list[i]
                S.remove(rem)
                best_add = -1
                for add in range(1, n_max + 1):
                    if add not in S and is_valid_addition(S, add):
                        best_add = add
                        break
                if best_add >= 0:
                    S.add(best_add)
                    improved = True
                else:
                    S.add(rem)
                if improved:
                    break
        return S

    def simulated_annealing(S, max_iters=10000):
        """Simulated annealing with 1-swap moves."""
        S = S.copy()
        best_S = S.copy()
        best_size = len(S)
        temp = 5.0
        cooling = 0.995
        iters = 0
        while iters < max_iters:
            if time.time() - start_time > deadline:
                break
            iters += 1
            # Random 1-swap: remove random element, add random element
            if not S:
                break
            rem = random.choice(list(S))
            S.remove(rem)
            candidates = [x for x in range(1, n_max + 1) if x not in S and is_valid_addition(S, x)]
            if candidates:
                add = random.choice(candidates)
                S.add(add)
                new_size = len(S)
                if new_size > best_size:
                    best_size = new_size
                    best_S = S.copy()
                else:
                    # Accept with probability exp(-delta/temp)
                    delta = best_size - new_size
                    if delta > 0 and random.random() < (2.718281828 ** (-delta / max(temp, 0.01))):
                        pass  # accept
                    else:
                        S.remove(add)
                        S.add(rem)
            else:
                S.add(rem)
            temp *= cooling
        return best_S

    # Generate different orders for greedy
    order_deg = list(range(1, n_max + 1))
    order_deg.sort(key=lambda x: triple_count[x])
    
    order_desc = list(range(n_max, 0, -1))
    order_asc = list(range(1, n_max + 1))
    
    best_set = set()
    best_size = 0
    
    # Run greedy with deterministic orders
    for order in [order_deg, order_desc, order_asc]:
        S = run_greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Run greedy with random orders
    for _ in range(30):
        if time.time() - start_time > deadline:
            break
        order_rand = list(range(1, n_max + 1))
        random.shuffle(order_rand)
        S = run_greedy(order_rand)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Apply local search refinement to best set
    if time.time() - start_time < deadline:
        refined = local_search_1swap(best_set, max_iters=3000)
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined
    
    # Try more random restarts with local search
    for _ in range(15):
        if time.time() - start_time > deadline:
            break
        order_rand = list(range(1, n_max + 1))
        random.shuffle(order_rand)
        S = run_greedy(order_rand)
        if len(S) >= best_size - 2:  # Only refine if close to best
            refined = local_search_1swap(S, max_iters=2000)
            if len(refined) > best_size:
                best_size = len(refined)
                best_set = refined
    
    # Final polish: try simulated annealing from best
    if time.time() - start_time < deadline - 5:
        sa_result = simulated_annealing(best_set, max_iters=5000)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
