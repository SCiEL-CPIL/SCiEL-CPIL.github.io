"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin (contract limit is 60s)
    
    N = 180
    
    # Precompute forbidden triples and conflict counts
    forbidden_triples = set()
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            denom = b - a
            num = a * b
            if num % denom != 0:
                continue
            c = num // denom
            if c <= b or c > N:
                continue
            triple = (a, b, c)
            forbidden_triples.add(triple)
            
    # Count conflicts for each number
    conflict_count = [0] * (N + 1)
    triples_with = [set() for _ in range(N + 1)]
    
    for t in forbidden_triples:
        a, b, c = t
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1
        
        triples_with[a].add((b, c))
        triples_with[b].add((a, c))
        triples_with[c].add((a, b))
        
    # Multiple ordering strategies
    orders = [
        sorted(range(1, N + 1), key=lambda x: (conflict_count[x], x)),
        sorted(range(1, N + 1), key=lambda x: (conflict_count[x], -x)),
        sorted(range(1, N + 1), key=lambda x: x),
        sorted(range(1, N + 1), key=lambda x: -x),
        sorted(range(1, N + 1), key=lambda x: (conflict_count[x] % 3, x)),
    ]
    
    def is_valid_addition(x, S_set):
        for u, v in triples_with[x]:
            if u in S_set and v in S_set:
                return False
        return True
        
    def greedy_order(order):
        S = set()
        for x in order:
            if is_valid_addition(x, S):
                S.add(x)
        return S
        
    def local_search(S, time_limit):
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add any missing element
            added_any = False
            for y in range(1, N + 1):
                if y not in S and is_valid_addition(y, S):
                    S.add(y)
                    improved = True
                    added_any = True
                    break
            if added_any:
                continue
                
            # 2. Try swaps: for each y not in S, find if removing some x from S allows adding y
            candidates_y = [y for y in range(1, N + 1) if y not in S]
            candidates_y.sort(key=lambda y: conflict_count[y])
            
            for y in candidates_y:
                if time.time() > time_limit:
                    break
                    
                # Find all x in S that block y (i.e., there exists a triple (y, u, v) with u,v in S and x in {u,v})
                blockers = set()
                for u, v in triples_with[y]:
                    if u in S and v in S:
                        blockers.add(u)
                        blockers.add(v)
                        
                if not blockers:
                    continue  # Shouldn't happen since we already checked is_valid_addition
                    
                # Try removing each blocker to see if y can be added
                for x in blockers:
                    S.remove(x)
                    if is_valid_addition(y, S):
                        S.add(y)
                        improved = True
                        break
                    S.add(x)  # Restore if it didn't work
                    
                if improved:
                    break
                    
        return S
        
    # Try all initial orders
    best_set = set()
    best_size = 0
    
    for order in orders:
        if time.time() > deadline - 5:
            break
        S = greedy_order(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            
    # Local search on best
    if time.time() < deadline - 5:
        S_improved = local_search(best_set, deadline - 5)
        if len(S_improved) > best_size:
            best_set = S_improved
            best_size = len(best_set)
            
    # Random restarts with perturbed orders
    base_order = orders[0]  # Primary order
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        # Create perturbed order
        order = list(base_order)
        # Randomly swap several pairs
        num_swaps = random.randint(3, 10)
        for _ in range(num_swaps):
            i, j = random.sample(range(len(order)), 2)
            order[i], order[j] = order[j], order[i]
        
        S_rand = greedy_order(order)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(S_rand)
            
            # Local search on new best
            if time.time() < deadline - 5:
                S_improved = local_search(best_set, deadline - 5)
                if len(S_improved) > best_size:
                    best_set = S_improved
                    best_size = len(best_set)
                    
    return sorted(best_set)
# EVOLVE-BLOCK-END
