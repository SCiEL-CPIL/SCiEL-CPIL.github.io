# EVOLVE-BLOCK-START
import time

N = 256
DEADLINE = 85  # 90s wall-clock limit, 5s safety margin

def construct():
    start_time = time.time()
    
    # Precompute 3-AP partners for each x
    # ap_partners[x] is a list of tuples (u, v) such that {x, u, v} is a 3-AP
    ap_partners = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # Case 1: x is the middle. a, x, b => b = 2x - a.
        # a < x < b.
        for a in range(1, x):
            b = 2 * x - a
            if b <= N and b > x:
                ap_partners[x].append((a, b))
        # Case 2: x is the largest. a, b, x => b = (a+x)/2.
        # a < b < x.
        for a in range(1, x):
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if a < b < x:
                    ap_partners[x].append((a, b))
        # Case 3: x is the smallest. x, a, b => b = 2a - x.
        # x < a < b.
        for a in range(x + 1, N + 1):
            b = 2 * a - x
            if b <= N and b > a:
                ap_partners[x].append((a, b))

    # Sort elements by 1/x descending to maximize bound early
    # We want to pick small numbers first.
    order = sorted(range(1, N + 1), key=lambda x: -1.0/x)
    
    # Precompute suffix sums of reciprocals for the upper bound
    # suffix_sum[i] = sum of 1/order[j] for j from i to N-1
    suffix_sum = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / order[i]
    
    best_sum = 0.0
    best_mask = 0
    best_set = []
    
    # B&B DFS
    def dfs(idx, current_sum, mask):
        nonlocal best_sum, best_mask, best_set
        
        # Pruning
        if current_sum + suffix_sum[idx] <= best_sum:
            return
        
        # Check time
        if time.time() - start_time > DEADLINE:
            return
        
        if idx == N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask = mask
                # Reconstruct set from mask
                s = []
                for x in range(1, N + 1):
                    if mask & (1 << (x - 1)):
                        s.append(x)
                best_set = s
            return
        
        x = order[idx]
        
        # Option 1: Include x
        # Check if x is compatible with current mask
        compatible = True
        for u, v in ap_partners[x]:
            if (mask & (1 << (u - 1))) and (mask & (1 << (v - 1))):
                compatible = False
                break
        
        if compatible:
            dfs(idx + 1, current_sum + 1.0 / x, mask | (1 << (x - 1)))
        
        # Option 2: Exclude x
        dfs(idx + 1, current_sum, mask)

    dfs(0, 0.0, 0)
    
    if not best_set:
        # Fallback: Greedy
        S = set()
        in_S = [False] * (N + 1)
        for x in range(1, N + 1):
            ok = True
            for a in S:
                if a == x: continue
                b1 = 2*x - a
                if 1 <= b1 <= N and in_S[b1] and b1 != a:
                    ok = False; break
                if (a+x) % 2 == 0:
                    b2 = (a+x)//2
                    if 1 <= b2 <= N and in_S[b2] and b2 != a and b2 != x:
                        ok = False; break
                b3 = 2*a - x
                if 1 <= b3 <= N and in_S[b3] and b3 != a and b3 != x:
                    ok = False; break
            if ok:
                S.add(x)
                in_S[x] = True
        return sorted(S)
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
