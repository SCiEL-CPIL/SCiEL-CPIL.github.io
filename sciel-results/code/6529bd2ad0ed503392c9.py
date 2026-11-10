import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    # Precompute conflicts
    N = 256
    conflicts = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i][j] = True
                conflicts[j][i] = True

    # Count conflicts for each number
    conflict_count = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if conflicts[i][j]:
                conflict_count[i] += 1
                conflict_count[j] += 1

    # Helper to check if x can be added to S
    def can_add(x, S):
        for y in S:
            if conflicts[x][y]:
                return False
        return True

    # Greedy build with given order
    def greedy_build(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # Fast local search: try swaps that increase size
    def local_search(S, max_iterations=2000):
        best_S = set(S)
        best_size = len(S)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try 1-for-1 swaps
            for x in list(best_S):
                if time.time() > deadline:
                    return best_S, best_size
                for y in range(1, N + 1):
                    if y in best_S:
                        continue
                    # Check if y conflicts with any element in best_S except x
                    valid = True
                    for z in best_S:
                        if z != x and conflicts[y][z]:
                            valid = False
                            break
                    if valid:
                        new_S = set(best_S)
                        new_S.remove(x)
                        new_S.add(y)
                        if len(new_S) > best_size:
                            best_S = new_S
                            best_size = len(new_S)
                            improved = True
                            break
                if improved:
                    break
            
            # Try 1-for-2 swaps: remove x, add y1, y2
            if not improved:
                for x in list(best_S):
                    if time.time() > deadline:
                        return best_S, best_size
                    candidates = [y for y in range(1, N + 1) if y not in best_S and conflicts[x][y]]
                    
                    for i in range(len(candidates)):
                        if time.time() > deadline:
                            return best_S, best_size
                        for j in range(i + 1, len(candidates)):
                            y1, y2 = candidates[i], candidates[j]
                            if conflicts[y1][y2]:
                                continue
                            valid = True
                            for z in best_S:
                                if z != x and (conflicts[y1][z] or conflicts[y2][z]):
                                    valid = False
                                    break
                            if valid:
                                new_S = set(best_S)
                                new_S.remove(x)
                                new_S.add(y1)
                                new_S.add(y2)
                                if len(new_S) > best_size:
                                    best_S = new_S
                                    best_size = len(new_S)
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        
        return best_S, best_size

    # Initial set: all odd numbers
    S = set(range(1, N + 1, 2))
    
    # Greedy addition of even numbers, sorted by conflict count
    evens = list(range(2, N + 1, 2))
    evens.sort(key=lambda x: conflict_count[x])
    
    for e in evens:
        if can_add(e, S):
            S.add(e)
    
    best_S = S
    best_size = len(S)
    
    # Local search
    best_S, best_size = local_search(S)
    
    # Multiple restarts
    for restart in range(300):
        if time.time() > deadline:
            break
        
        all_numbers = list(range(1, N + 1))
        all_numbers.sort(key=lambda x: conflict_count[x] + random.random() * 5)
        
        S_current = greedy_build(all_numbers)
        S_current, size = local_search(S_current, max_iterations=500)
        
        if size > best_size:
            best_size = size
            best_S = S_current
    
    return list(best_S)
