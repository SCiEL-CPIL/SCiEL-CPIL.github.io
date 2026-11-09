import random
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115
    N = 284
    PRESCRIBED = [1, 2, 5, 11]
    PRESCRIBED_SET = set(PRESCRIBED)
    
    def get_sums_set(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def count_collisions(S):
        sums = []
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.append(S[i] + S[j])
        return len(sums) - len(set(sums))

    def score(S):
        c = count_collisions(S)
        if c > 0:
            return len(S) - 100 * c
        return len(S)

    def is_valid(S):
        return count_collisions(S) == 0

    # Improved greedy initialization
    def greedy_init():
        S = list(PRESCRIBED)
        S_set = set(S)
        sums_set = get_sums_set(S)
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in S_set:
                continue
            # Check if adding x creates any collision
            new_sums = []
            valid = True
            for y in S:
                s = x + y
                if s in sums_set:
                    valid = False
                    break
                new_sums.append(s)
            if valid:
                S.append(x)
                S_set.add(x)
                sums_set.update(new_sums)
        
        return S

    # Local polish: try to fix collisions by moving individual elements
    def local_polish(S, max_iterations=500):
        S = list(S)
        S_set = set(S)
        best_score = score(S)
        best_S = list(S)
        
        for _ in range(max_iterations):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if is_valid(S):
                return S
            
            # Find elements involved in collisions
            sums = {}
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s = S[i] + S[j]
                    if s in sums:
                        # Collision found
                        # Try moving one of the elements
                        for elem_idx in [i, j]:
                            if S[elem_idx] in PRESCRIBED_SET:
                                continue
                            old_val = S[elem_idx]
                            # Try replacing with a random candidate
                            for _ in range(20):
                                new_val = random.randint(1, N)
                                if new_val in S_set:
                                    continue
                                if new_val == old_val:
                                    continue
                                # Test replacement
                                S[elem_idx] = new_val
                                S_set.discard(old_val)
                                S_set.add(new_val)
                                new_score = score(S)
                                if new_score > best_score:
                                    best_score = new_score
                                    best_S = list(S)
                                # Revert
                                S[elem_idx] = old_val
                                S_set.discard(new_val)
                                S_set.add(old_val)
                    else:
                        sums[s] = (i, j)
            
            S = best_S
            S_set = set(S)
        
        return best_S

    # Multiple restarts for better initialization
    best_S = None
    best_score = -1
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        S = greedy_init()
        S = local_polish(S, max_iterations=200)
        s = score(S)
        if s > best_score:
            best_score = s
            best_S = list(S)
    
    # Simulated Annealing on the best solution
    T = 5.0
    T_min = 0.001
    alpha = 0.99995
    
    current_S = list(best_S)
    current_score = score(current_S)
    
    no_improve_count = 0
    patience = 5000
    
    while time.time() - start_time < DEADLINE:
        # Generate neighbor
        move_type = random.random()
        new_S = list(current_S)
        new_S_set = set(new_S)
        
        if move_type < 0.3:
            # Add
            while True:
                x = random.randint(1, N)
                if x not in new_S_set:
                    break
            new_S.append(x)
        elif move_type < 0.7:
            # Swap
            non_prescribed = [y for y in new_S if y not in PRESCRIBED_SET]
            if not non_prescribed:
                continue
            y = random.choice(non_prescribed)
            new_S.remove(y)
            new_S_set.discard(y)
            
            while True:
                x = random.randint(1, N)
                if x not in new_S_set:
                    break
            new_S.append(x)
            new_S_set.add(x)
        else:
            # Remove
            non_prescribed = [y for y in new_S if y not in PRESCRIBED_SET]
            if not non_prescribed:
                continue
            y = random.choice(non_prescribed)
            new_S.remove(y)
            new_S_set.discard(y)
            
        new_score = score(new_S)
        delta = new_score - current_score
        
        if delta > 0 or (T > T_min and random.random() < math.exp(delta / T)):
            current_S = new_S
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_S = list(current_S)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= alpha
        if T < T_min:
            T = T_min
        
        # If stuck for too long, try a small perturbation
        if no_improve_count > patience:
            # Do a few random swaps to escape local minimum
            for _ in range(10):
                if time.time() - start_time > DEADLINE - 5:
                    break
                non_prescribed = [y for y in current_S if y not in PRESCRIBED_SET]
                if not non_prescribed:
                    break
                idx = random.randint(0, len(current_S) - 1)
                if current_S[idx] in PRESCRIBED_SET:
                    continue
                old_val = current_S[idx]
                new_val = random.randint(1, N)
                if new_val in set(current_S) or new_val == old_val:
                    continue
                current_S[idx] = new_val
                new_score = score(current_S)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = list(current_S)
            no_improve_count = 0
    
    # Final local polish
    best_S = local_polish(best_S, max_iterations=1000)
    
    return best_S
