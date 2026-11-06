import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # 1. Greedy Selection
    chosen = []
    # Precompute gcds? Or just use math.gcd
    for a in range(511, 0, -1):
        if all(math.gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    def get_score(S):
        return sum(1.0 / (512 - a) for a in S)
    
    current_score = get_score(chosen)
    best_set = chosen
    best_score = current_score
    
    # 2. Local Search
    # We will try to improve the set by swapping or replacing.
    # Since we want to maximize sum, and weights are decreasing, 
    # we focus on replacing smaller elements with larger compatible ones.
    
    # Optimization: Precompute compatibility matrix? 
    # 511x511 boolean matrix is ~250k bytes. Very fast to build.
    compat = [[False]*512 for _ in range(512)]
    for i in range(1, 512):
        for j in range(i+1, 512):
            if math.gcd(i, j) == 1:
                compat[i][j] = True
                compat[j][i] = True
    # Diagonal is False (distinct integers)
    
    # Helper to check if x is compatible with set S
    def is_compat_with_set(x, S):
        return all(compat[x][y] for y in S)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try single swaps: replace a in S with b not in S
        # b should be > a to potentially improve score (since w is increasing in a)
        # But actually w(b) > w(a) is the condition.
        for i, a in enumerate(best_set):
            for b in range(511, a, -1):
                if b in best_set:
                    continue
                # Check if b is compatible with S \ {a}
                if all(compat[b][y] for y in best_set if y != a):
                    new_score = best_score - 1.0/(512-a) + 1.0/(512-b)
                    if new_score > best_score + 1e-9:
                        best_set[i] = b
                        best_score = new_score
                        improved = True
                        break # Restart loop as set changed
            if improved:
                break
        
        # Try double replacement: replace a in S with {b, d} not in S
        # This is more expensive.
        # We can limit this to cases where a is relatively small or we have time.
        # Given the time budget, single swaps are likely sufficient.
        # But let's add a limited double swap check if time permits.
        
    return best_set
