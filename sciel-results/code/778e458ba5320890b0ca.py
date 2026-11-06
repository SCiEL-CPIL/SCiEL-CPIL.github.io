import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin within 90s
    
    N = 256
    # Start with all odd numbers (128 elements)
    S = set(range(1, N + 1, 2))
    
    # Precompute conflict matrix: conflict[a][b] = True if (a*b)%(a+b)==0
    # For a, b in [1, 256]
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a][b] = True
                conflict[b][a] = True
    
    # Helper: check if x can be added to set S using conflict matrix
    def can_add(x, S):
        for y in S:
            if conflict[x][y]:
                return False
        return True
    
    # Helper: count conflicts between x and set S
    def count_conflicts(x, S):
        cnt = 0
        for y in S:
            if conflict[x][y]:
                cnt += 1
        return cnt
    
    # Greedy add in a given order
    def greedy_add(evens, S_init):
        S = set(S_init)
        for e in evens:
            if can_add(e, S):
                S.add(e)
        return S
    
    # Local search: try removing elements and re-adding
    def local_search(S, evens, max_rounds=10):
        current_S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 2:
            improved = False
            rounds += 1
            for y in list(current_S):
                if time.time() > deadline - 2:
                    break
                test_S = current_S - {y}
                # Try to add evens not in test_S
                added = 0
                for e in evens:
                    if e in test_S:
                        continue
                    if can_add(e, test_S):
                        test_S.add(e)
                        added += 1
                if added > 1:  # net gain
                    current_S = test_S
                    improved = True
                    break
        return current_S
    
    # Simulated annealing
    def anneal(S_init, evens, steps=5000, T_start=10.0, T_end=0.01):
        S = set(S_init)
        best_S = set(S)
        best_size = len(S)
        T = T_start
        cooling = T_start / max(1, steps)
        
        for step in range(steps):
            if time.time() > deadline - 3:
                break
            T = max(T_end, T_start * (T_end / T_start) ** (step / max(1, steps)))
            
            # Random move: add, remove, or swap
            move_type = random.random()
            new_S = set(S)
            
            if move_type < 0.4:
                # Try to add a random element not in S
                candidates = [x for x in range(1, N + 1) if x not in S]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_S.add(x)
            elif move_type < 0.7:
                # Try to remove a random element from S
                if not S:
                    continue
                x = random.choice(list(S))
                new_S.discard(x)
            else:
                # Swap: remove one, add another
                if not S:
                    continue
                x = random.choice(list(S))
                candidates = [y for y in range(1, N + 1) if y not in S]
                if not candidates:
                    continue
                y = random.choice(candidates)
                new_S.discard(x)
                new_S.add(y)
            
            # Check validity of new_S
            valid = True
            s_list = list(new_S)
            for i in range(len(s_list)):
                for j in range(i + 1, len(s_list)):
                    if conflict[s_list[i]][s_list[j]]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Revert
                continue
            
            # Accept if better or with probability exp(-delta/T)
            delta = len(new_S) - len(S)
            if delta >= 0:
                S = new_S
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
            else:
                if random.random() < math.exp(delta / max(T, 1e-10)):
                    S = new_S
        
        return best_S
    
    # Multiple initializations
    best_S = set(S)
    best_size = len(S)
    
    evens = list(range(2, N + 1, 2))
    
    # Order 1: natural
    S1 = greedy_add(evens, S)
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
    
    # Order 2: reversed
    S2 = greedy_add(reversed(evens), S)
    if len(S2) > best_size:
        best_S = S2
        best_size = len(S2)
    
    # Order 3: several random shuffles
    for trial in range(8):
        if time.time() > deadline - 5:
            break
        evens_shuffled = evens[:]
        random.shuffle(evens_shuffled)
        S3 = greedy_add(evens_shuffled, S)
        if len(S3) > best_size:
            best_S = S3
            best_size = len(S3)
    
    # Local search on best
    if time.time() < deadline - 5:
        best_S = local_search(best_S, evens, max_rounds=15)
        best_size = len(best_S)
    
    # Simulated annealing refinement
    if time.time() < deadline - 5:
        annealed = anneal(best_S, evens, steps=8000, T_start=15.0, T_end=0.005)
        if len(annealed) > best_size:
            best_S = annealed
            best_size = len(best_S)
    
    # Final local search
    if time.time() < deadline - 2:
        best_S = local_search(best_S, evens, max_rounds=10)
        best_size = len(best_S)
    
    return list(best_S)
