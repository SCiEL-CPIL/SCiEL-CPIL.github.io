import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 55.0  # Safety margin

    # Pre-compute degrees for heuristic
    # Degree of x is number of 4-APs containing x
    degrees = [0] * (n + 1)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            x1, x2, x3, x4 = a, a + d, a + 2 * d, a + 3 * d
            degrees[x1] += 1
            degrees[x2] += 1
            degrees[x3] += 1
            degrees[x4] += 1

    def can_add(in_S, u):
        # in_S is a list of bools, 1-indexed
        # Check 4 cases
        
        # Case 1: u is a (start)
        # u, u+d, u+2d, u+3d
        max_d1 = (n - u) // 3
        for d in range(1, max_d1 + 1):
            if in_S[u + d] and in_S[u + 2 * d] and in_S[u + 3 * d]:
                return False
        
        # Case 2: u is a+d (2nd)
        # u-d, u, u+d, u+2d
        # Need u-d, u+d, u+2d in S
        max_d2 = (n - u) // 2
        if u - 1 > 0:
            max_d2 = min(max_d2, u - 1)
        for d in range(1, max_d2 + 1):
            if in_S[u - d] and in_S[u + d] and in_S[u + 2 * d]:
                return False
        
        # Case 3: u is a+2d (3rd)
        # u-2d, u-d, u, u+d
        # Need u-2d, u-d, u+d in S
        max_d3 = n - u
        if u - 1 >= 2:
            max_d3 = min(max_d3, (u - 1) // 2)
        for d in range(1, max_d3 + 1):
            if in_S[u - 2 * d] and in_S[u - d] and in_S[u + d]:
                return False
        
        # Case 4: u is a+3d (4th)
        # u-3d, u-2d, u-d, u
        # Need u-3d, u-2d, u-d in S
        max_d4 = (u - 1) // 3
        for d in range(1, max_d4 + 1):
            if in_S[u - 3 * d] and in_S[u - 2 * d] and in_S[u - d]:
                return False
        
        return True

    def greedy_init(strategy="degree"):
        in_S = [False] * (n + 1)
        S_count = 0
        
        if strategy == "degree":
            candidates = list(range(1, n + 1))
            candidates.sort(key=lambda x: degrees[x])
        elif strategy == "random":
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        else: # mixed
            candidates = list(range(1, n + 1))
            # Sort by degree, then randomize ties? Or just random shuffle weighted by degree?
            # Let's stick to simple random or degree for now.
            random.shuffle(candidates)

        for u in candidates:
            if can_add(in_S, u):
                in_S[u] = True
                S_count += 1
        
        return in_S, S_count

    def local_search(in_S, S_count):
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add any available element (1-0 swap)
            # This is usually empty if greedy was maximal, but after swaps it might not be.
            added_any = False
            for u in range(1, n + 1):
                if not in_S[u]:
                    if can_add(in_S, u):
                        in_S[u] = True
                        S_count += 1
                        added_any = True
                        improved = True
                        # Restart loop to re-evaluate
                        break
            if added_any:
                continue

            # 2. Try 2-1 swaps
            # Remove v, add u1, u2
            S_list = [i for i in range(1, n + 1) if in_S[i]]
            random.shuffle(S_list)
            
            for v in S_list:
                if time.time() - start_time > time_limit:
                    break
                
                # Temporarily remove v
                in_S[v] = False
                S_count -= 1
                
                # Find 2 elements that can be added
                added = []
                for u in range(1, n + 1):
                    if in_S[u]:
                        continue
                    if can_add(in_S, u):
                        added.append(u)
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    # Accept swap
                    in_S[added[0]] = True
                    in_S[added[1]] = True
                    S_count += 2
                    improved = True
                    # Break to restart while loop
                    break
                else:
                    # Revert
                    in_S[v] = True
                    S_count += 1
        
        return in_S, S_count

    best_in_S = None
    best_count = 0

    # Run multiple strategies/restarts
    strategies = ["degree", "random", "random", "random"]
    
    # First, try a few degree-based greedies
    for _ in range(5):
        in_S, count = greedy_init("degree")
        in_S, count = local_search(in_S, count)
        if count > best_count:
            best_count = count
            best_in_S = in_S[:]
        
        if time.time() - start_time > time_limit:
            break

    # Then, try random restarts
    while time.time() - start_time < time_limit:
        in_S, count = greedy_init("random")
        in_S, count = local_search(in_S, count)
        if count > best_count:
            best_count = count
            best_in_S = in_S[:]

    if best_in_S is None:
        # Fallback to simple greedy
        in_S, count = greedy_init("degree")
        best_in_S = in_S
        best_count = count

    return [i for i in range(1, n + 1) if best_in_S[i]]

