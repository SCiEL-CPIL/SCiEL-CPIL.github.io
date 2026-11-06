import time
import random

def construct():
    start_time = time.time()
    time_limit = 60.0
    N = 256
    
    inv = [0.0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = 1.0 / i

    def can_add(x, A, exclude):
        # Check if x can be added to A with 'exclude' removed
        # A is a set
        # x is not in A
        
        # Case 1 & 2: x is end or start.
        for b in A:
            if b == exclude:
                continue
            # x is 3rd (a, b, x) => a = 2b - x
            a = 2 * b - x
            if 0 < a < b:
                if a != exclude and a in A:
                    return False
            # x is 1st (x, b, c) => c = 2b - x
            c = 2 * b - x
            if c > b and c <= N:
                if c != exclude and c in A:
                    return False
        
        # Case 3: x is middle (a, x, c) => a + c = 2x
        for a in A:
            if a == exclude:
                continue
            c = 2 * x - a
            if c > x and c <= N:
                if c != exclude and c in A:
                    return False
        return True

    def get_score(A):
        s = 0.0
        for x in A:
            s += inv[x]
        return s

    def greedy_initial():
        A = set()
        for x in range(1, N + 1):
            if can_add(x, A, -1):
                A.add(x)
        return A

    def greedy_random():
        A = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, A, -1):
                A.add(x)
        return A

    best_set = greedy_initial()
    best_score = get_score(best_set)
    
    current_set = set(best_set)
    current_score = best_score

    while time.time() - start_time < time_limit:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in current_set:
            for j in range(1, N + 1):
                if j in current_set:
                    continue
                
                if can_add(j, current_set, i):
                    delta = inv[j] - inv[i]
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
        
        if best_delta > 1e-9:
            current_set.remove(best_i)
            current_set.add(best_j)
            current_score += best_delta
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            perturbed = False
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                i = random.choice(list(current_set))
                j = random.randint(1, N)
                if j in current_set:
                    continue
                if can_add(j, current_set, i):
                    delta = inv[j] - inv[i]
                    if delta > -0.2:
                        current_set.remove(i)
                        current_set.add(j)
                        current_score += delta
                        perturbed = True
                        break
            
            if not perturbed:
                if random.random() < 0.5:
                    current_set = greedy_initial()
                else:
                    current_set = greedy_random()
                current_score = get_score(current_set)
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)

    return list(best_set)
