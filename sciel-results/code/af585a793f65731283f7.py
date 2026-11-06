"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85
    MOD = 1009
    N = MOD - 1  # 1008

    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    # Identify self-inverse elements
    self_inv = set()
    for x in range(1, MOD):
        if (x * x) % MOD == 1:
            self_inv.add(x)
    
    # Precompute next_self_inv[s]: smallest self-inverse element >= s
    # If none, use N+1 (infinity for our bounds)
    next_self_inv = [N + 1] * (N + 2)
    current_next = N + 1
    for s in range(N, 0, -1):
        if s in self_inv:
            current_next = s
        next_self_inv[s] = current_next

    # Precompute pair_b_at_a[a]
    # For each pair {x, inv[x]} with x < inv[x], let a=x, b=inv[x].
    # If x > inv[x], let a=inv[x], b=x.
    # We only need to store this for the smaller element a.
    pair_b_at_a = [N + 1] * (N + 2)
    visited = [False] * MOD
    for x in range(1, MOD):
        if visited[x]:
            continue
        ix = inv[x]
        if ix == x:
            continue # Self-inverse, handled separately
        a = min(x, ix)
        b = max(x, ix)
        pair_b_at_a[a] = b
        visited[x] = True
        visited[ix] = True

    best_start = 1
    best_len = 0
    current_min_b = N + 1

    for s in range(N, 0, -1):
        if time.time() > deadline:
            break
        
        # Update current_min_b with the pair starting at s
        if pair_b_at_a[s] < current_min_b:
            current_min_b = pair_b_at_a[s]
            
        # The interval [s, e] is valid if:
        # 1. e < current_min_b (no full pair contained)
        # 2. e < next_self_inv[s] (no self-inverse contained)
        
        limit = min(current_min_b, next_self_inv[s])
        
        # Max valid end is limit - 1
        # Length is (limit - 1) - s + 1 = limit - s
        length = limit - s
        
        if length > best_len:
            best_len = length
            best_start = s

    if best_len == 0:
        return [2, 1]
        
    return [best_start, best_len]
# EVOLVE-BLOCK-END
