import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter: only divisors <= target can be used
    divisors = [d for d in divisors if d <= target]
    divisors.sort()
    
    N = len(divisors)
    
    # Quick greedy to get an upper bound on k
    remainder = target
    greedy_chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            greedy_chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    greedy_k = len(greedy_chosen) if remainder == 0 else N + 1
    
    # DP: dp[s] = min number of divisors to sum to s, using a 1D array with backtracking
    # We need to track which divisors were used for backtracking
    # Use a 2D approach: dp[i][s] = min count using first i divisors to sum to s
    # But store parent pointers for reconstruction
    
    INF = 10**9
    # dp[i][s] = minimum number of divisors from divisors[0..i-1] to sum to s
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    # parent[i][s] = whether we took divisor i-1 to achieve dp[i][s]
    parent = [[False] * (target + 1) for _ in range(N + 1)]
    
    dp[0][0] = 0
    
    for i in range(1, N + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't take divisor i
            val1 = dp[i-1][s]
            take1 = False
            
            # Option 2: Take divisor i (if s >= d)
            val2 = INF
            take2 = False
            if s >= d and dp[i-1][s-d] != INF:
                val2 = dp[i-1][s-d] + 1
                take2 = True
            
            if val1 <= val2:
                dp[i][s] = val1
                parent[i][s] = False
            else:
                dp[i][s] = val2
                parent[i][s] = True
        
        # Time check
        if time.time() > deadline:
            break
    
    # Check if solution exists
    if dp[N][target] == INF:
        # Fallback: return greedy result if valid
        if remainder == 0:
            return greedy_chosen
        return []
    
    # Backtrack to find the divisors
    selected = []
    s = target
    for i in range(N, 0, -1):
        if parent[i][s]:
            selected.append(divisors[i-1])
            s -= divisors[i-1]
    
    # Verify
    assert sum(selected) == target, f"Sum mismatch: {sum(selected)} != {target}"
    assert len(set(selected)) == len(selected), "Not distinct"
    for d in selected:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    
    return selected
