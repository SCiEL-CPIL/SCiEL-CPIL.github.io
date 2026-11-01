import time

def get_overlaps(perms):
    n_perms = len(perms)
    n_len = len(perms[0])
    overlaps = [[0] * n_perms for _ in range(n_perms)]
    for i in range(n_perms):
        p_i = perms[i]
        for j in range(n_perms):
            p_j = perms[j]
            for k in range(n_len - 1, 0, -1):
                if p_i[-k:] == p_j[:k]:
                    overlaps[i][j] = k
                    break
    return overlaps

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0

    if n == 1:
        return "1"

    # Build S_{n-1}
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new

    # Extract perms of n-1
    perms, seen = [], set()
    for i in range(len(word) - (n - 1) + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)

    N = len(perms)
    
    if N < 100:
        # Small enough for exhaustive or heavy search?
        # For n=5, N=24. n=6, N=120.
        # For n=7, N=720.
        pass

    overlaps = get_overlaps(perms)
    
    # Initial order
    order = list(range(N))
    
    def calc_cost(order):
        cost = 0
        for i in range(len(order) - 1):
            cost += overlaps[order[i]][order[i+1]]
        return cost

    current_cost = calc_cost(order)
    
    # 2-Opt Local Search
    # We will perform passes of 2-opt until no improvement or time limit.
    # To speed up, we can limit the search space or use a smarter check.
    # Given Python, O(N^3) 2-opt is risky. 
    # Let's use a "Greedy 2-Opt": for each i, find the best j to swap/reverse?
    # Or just simple adjacent swaps (2-opt with len=1) and maybe len=2?
    
    # Let's try a simple hill-climbing with random restarts if time permits?
    # No, deterministic is safer.
    
    # Let's implement a basic 2-opt with a time limit per iteration.
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline - 10:
                break
            for j in range(i + 2, N):
                if time.time() > deadline - 10:
                    break
                # Check if reversing order[i+1:j+1] improves cost
                # This is expensive to check exactly without prefix sums.
                # Let's stick to Adjacent Swaps for safety and speed.
                # Adjacent swap is i, j=i+1
                if j != i + 1:
                    continue
                
                u = order[i]
                v = order[i+1]
                
                prev = order[i-1] if i > 0 else -1
                next = order[i+2] if i+2 < N else -1
                
                old_cost = 0
                new_cost = 0
                
                if prev != -1:
                    old_cost += overlaps[prev][u]
                    new_cost += overlaps[prev][v]
                
                old_cost += overlaps[u][v]
                new_cost += overlaps[v][u]
                
                if next != -1:
                    old_cost += overlaps[v][next]
                    new_cost += overlaps[u][next]
                
                if new_cost > old_cost:
                    order[i], order[i+1] = order[i+1], order[i]
                    current_cost += (new_cost - old_cost)
                    improved = True
                    break # Restart sweep
        
    # Construct final
    new = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
        new += block[overlap:]
        
    return "".join(map(str, new))
