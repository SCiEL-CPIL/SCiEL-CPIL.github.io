import time

def construct():
    """
    Finds the minimum number of distinct divisors of 5040 that sum to 4979.
    Uses Dynamic Programming to minimize the count of elements for each sum.
    """
    N = 5040
    TARGET = 4979
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(N ** 0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i * i != N:
                divisors.append(N // i)
    
    # Sort divisors. Order doesn't strictly matter for 0/1 DP correctness,
    # but sorting can help with cache locality or early pruning if we added such checks.
    # Here we just need the list.
    divisors.sort()
    
    # DP arrays
    # dp[s] = minimum number of divisors required to sum to s
    # parent[s] = (previous_sum, divisor_used) to reconstruct the set
    INF = float('inf')
    dp_count = [INF] * (TARGET + 1)
    dp_parent = [(-1, -1)] * (TARGET + 1)
    
    dp_count[0] = 0
    
    # Iterate through each divisor
    for d in divisors:
        if d > TARGET:
            continue
        # Update in reverse to ensure each divisor is used at most once
        # We iterate from TARGET down to d
        for s in range(TARGET, d - 1, -1):
            # If we can form s-d with fewer items, and adding d gives fewer items for s
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_parent[s] = (s - d, d)
    
    # Check if solution exists
    if dp_count[TARGET] == INF:
        # Fallback: This should not happen for this specific instance,
        # but we return a greedy solution to avoid crashing if logic fails unexpectedly.
        # However, the problem guarantees a solution exists.
        rem = TARGET
        chosen = []
        # Greedy largest first
        for d in sorted(divisors, reverse=True):
            if d <= rem:
                chosen.append(d)
                rem -= d
        if rem != 0:
            raise RuntimeError("No solution found in DP, greedy fallback failed")
        return chosen

    # Reconstruct the solution
    chosen = []
    current_sum = TARGET
    while current_sum > 0:
        prev_sum, d = dp_parent[current_sum]
        chosen.append(d)
        current_sum = prev_sum
        
    # The reconstruction gives the divisors in reverse order of addition (or arbitrary),
    # but the set is what matters.
    
    # Sanity checks
    # 1. Distinctness
    assert len(set(chosen)) == len(chosen), "Divisors are not distinct"
    # 2. Sum
    assert sum(chosen) == TARGET, f"Sum is {sum(chosen)}, expected {TARGET}"
    # 3. Divisor property
    assert all(N % d == 0 for d in chosen), "Selected numbers are not divisors of N"
    # 4. Count limit
    assert len(chosen) <= 256, "Too many divisors selected"
    
    return chosen

if __name__ == "__main__":
    start_time = time.time()
    result = construct()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.4f}s")
    print(f"Count: {len(result)}")
    print(f"Sum: {sum(result)}")
    print(f"Divisors: {sorted(result)}")
