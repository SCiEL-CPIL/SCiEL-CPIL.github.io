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
    for _ in range(50):
        if time.time() - start_time > deadline:
            break
        order_rand = list(range(1, n_max + 1))
        random.shuffle(order_rand)
        S = run_greedy(order_rand)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Local search: try to improve by removing one element and adding another
    def local_search(S, max_iterations=5000):
        S = S.copy()
        current_size = len(S)
        best_local = S.copy()
        best_local_size = current_size
        
        # Precompute for each element x, the pairs (y,z) that would form a forbidden triple if x were added
        # conflicts[x] contains (y, z) such that {x, y, z} is forbidden
        # To check if we can add x: no (y,z) in conflicts[x] with both y,z in S
        
        # For removal: if we remove x, we can potentially add some y not in S
        # such that adding y doesn't create a forbidden triple
        
        for iteration in range(max_iterations):
            if time.time() - start_time > deadline:
                break
            
            # Try random swaps: remove one element, try to add another
            if not S:
                break
                
            removed = random.choice(list(S))
            S.remove(removed)
            
            # Try to find an element to add back (possibly different from removed)
            candidates = [x for x in range(1, n_max + 1) if x not in S]
            random.shuffle(candidates)
            
            added = None
            for cand in candidates:
                if is_valid_addition(S, cand):
                    S.add(cand)
                    added = cand
                    break
            
            if added is not None and len(S) > best_local_size:
                best_local_size = len(S)
                best_local = S.copy()
            elif added is None:
                # Put removed back
                S.add(removed)
            elif len(S) < best_local_size:
                # This swap didn't help, revert
                S.remove(added)
                S.add(removed)
        
        return best_local
    
    # Run local search from best greedy solution
    if time.time() - start_time < deadline - 5:
        improved = local_search(best_set, max_iterations=10000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved.copy()
    
    # Run multiple rounds of greedy + local search
    for round_num in range(10):
        if time.time() - start_time > deadline - 10:
            break
        
        # Get a new greedy solution
        order_rand = list(range(1, n_max + 1))
        random.shuffle(order_rand)
        S = run_greedy(order_rand)
        
        # Apply local search
        if time.time() - start_time < deadline - 5:
            improved = local_search(S, max_iterations=5000)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved.copy()
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
