import time

def construct():
    N = 128
    
    # Try to find a coloring with minimal number of colors k
    # Start from 3 (theoretical minimum for this constraint)
    # Up to 10 or until time runs out
    global_start = time.time()
    
    for k in range(3, 11):
        if time.time() - global_start > 45:
            break
            
        start_time = time.time()
        timeout = 5.0
        solution = solve_k(k, N, timeout)
        if solution is not None:
            return solution
            
    # Fallback to greedy construction if no small-k solution found in time
    return greedy_construct(N)

def solve_k(k, N, timeout):
    colors = [0] * (N + 1)
    
    # Precompute the differences d for each position x that form a 4-term progression ending at x
    # Progression: x-3d, x-2d, x-d, x
    # Condition: x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3
    progs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            progs[x].append(d)
            
    def backtrack(x):
        if x > N:
            return True
        if time.time() > timeout:
            return False
            
        for c in range(k):
            colors[x] = c
            valid = True
            for d in progs[x]:
                c1 = colors[x - 3*d]
                c2 = colors[x - 2*d]
                c3 = colors[x - d]
                # Check if the set of colors has size at least 3
                if len({c1, c2, c3, c}) < 3:
                    valid = False
                    break
            if valid:
                if backtrack(x + 1):
                    return True
        return False

    if backtrack(1):
        return colors[1:]
    return None

def greedy_construct(N):
    colors = [0] * (N + 1)
    for x in range(1, N + 1):
        for c in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                c1 = colors[x - 3*d]
                c2 = colors[x - 2*d]
                c3 = colors[x - d]
                if len({c1, c2, c3, c}) < 3:
                    valid = False
                    break
            if valid:
                colors[x] = c
                break
    return colors[1:]
