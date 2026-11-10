"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 180
    
    # Precompute x_c_map[x][b] = c if 1/x = 1/b + 1/c with valid distinct c
    x_c_map = {}
    for x in range(1, N+1):
        x_c_map[x] = {}
        for b in range(1, N+1):
            if b == x:
                continue
            denom = b - x
            if denom == 0:
                continue
            num = x * b
            if num % denom != 0:
                continue
            c = num // denom
            if 1 <= c <= N and c != x and c != b:
                x_c_map[x][b] = c
    
    def is_valid_addition(x, S_set):
        for b in S_set:
            if b == x:
                continue
            c = x_c_map[x].get(b)
            if c is not None and c in S_set:
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
            # Try single swaps: remove x, add y
            for y in range(1, N+1):
                if y in S:
                    continue
                for x in list(S):
                    S.remove(x)
                    if is_valid_addition(y, S):
                        S.add(y)
                        improved = True
                        break
                    else:
                        S.add(x)
                if improved:
                    break
        return S
    
    # Multiple greedy strategies
    best_set = set()
    best_size = 0
    
    # Natural order
    S = greedy_order(range(1, N+1))
    if len(S) > best_size:
        best_set, best_size = S, len(S)
    
    # Reverse order
    S = greedy_order(range(N, 0, -1))
    if len(S) > best_size:
        best_set, best_size = S, len(S)
    
    # Random orders
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        S = greedy_order(order)
        if len(S) > best_size:
            best_set, best_size = S, len(S)
    
    # Local search on best
    if time.time() < deadline - 5:
        S_improved = local_search(best_set, deadline - 5)
        if len(S_improved) > best_size:
            best_set, best_size = S_improved, len(S_improved)
    
    # More random restarts with local search if time permits
    remaining = deadline - time.time()
    if remaining > 20:
        for trial in range(30):
            if time.time() > deadline - 10:
                break
            order = list(range(1, N+1))
            rng.shuffle(order)
            S = greedy_order(order)
            if len(S) >= best_size - 2:  # only local search promising starts
                S_improved = local_search(S, deadline - 5)
                if len(S_improved) > best_size:
                    best_set, best_size = S_improved, len(S_improved)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
