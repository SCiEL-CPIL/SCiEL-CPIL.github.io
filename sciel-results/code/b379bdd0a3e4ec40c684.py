import time
import sys

# EVOLVE-BLOCK-START
def construct():
    N = 256
    # Precompute suffix sums of reciprocals for upper bound
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / i
    
    # Initial solution using greedy
    def greedy_construct():
        selected = []
        mask = 0
        blocked = 0
        for x in range(1, N + 1):
            if not (blocked >> (x - 1)) & 1:
                selected.append(x)
                mask |= (1 << (x - 1))
                # Update blocked
                for y in selected[:-1]:
                    z = 2 * x - y
                    if x < z <= N:
                        blocked |= (1 << (z - 1))
        return selected, mask, blocked

    best_set, best_mask, _ = greedy_construct()
    best_score = sum(1.0 / x for x in best_set)
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    sys.setrecursionlimit(3000)
    
    def dfs(x, mask, blocked, score, selected):
        nonlocal best_score, best_set, best_mask, deadline
        
        if time.time() > deadline:
            return

        if score > best_score:
            best_score = score
            best_set = selected[:]
            best_mask = mask

        if x > N:
            return

        if score + suffix_sum[x] <= best_score:
            return

        # Option 1: Include x
        if not (blocked >> (x - 1)) & 1:
            new_mask = mask | (1 << (x - 1))
            new_blocked = blocked
            selected.append(x)
            
            # Update blocked for new x
            # We iterate over previous elements in selected
            # selected[-1] is x, so selected[:-1] are < x
            for y in selected[:-1]:
                z = 2 * x - y
                if x < z <= N:
                    new_blocked |= (1 << (z - 1))
            
            dfs(x + 1, new_mask, new_blocked, score + 1.0 / x, selected)
            selected.pop()
        
        # Option 2: Exclude x
        dfs(x + 1, mask, blocked, score, selected)

    dfs(1, 0, 0, 0.0, [])

    return best_set
# EVOLVE-BLOCK-END
