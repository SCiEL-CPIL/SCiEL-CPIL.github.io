import time
import random
import math

def construct():
    N = 256
    # Precomputation
    incidents = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N:
                continue
            incidents[a].append((b, c))
            incidents[b].append((a, c))
            incidents[c].append((a, b))
            
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    in_S = [False] * (N + 1)
    bad_count = [0] * (N + 1)
    current_score = 0.0
    size = 0
    
    def add(x):
        nonlocal in_S, bad_count, current_score, size
        in_S[x] = True
        current_score += weights[x]
        size += 1
        for y, z in incidents[x]:
            if in_S[z]:
                bad_count[y] += 1
            
    def remove(x):
        nonlocal in_S, bad_count, current_score, size
        in_S[x] = False
        current_score -= weights[x]
        size -= 1
        for y, z in incidents[x]:
            if in_S[z]:
                bad_count[y] -= 1
                
    # Greedy Initialization
    for i in range(1, N + 1):
        if bad_count[i] == 0:
            add(i)
            
    best_score = current_score
    best_set = in_S[:]
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 80.0
    
    T = 0.1
    alpha = 0.999
    min_T = 1e-6
    
    indices = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit:
        r = random.random()
        
        if r < 0.5:
            x = random.choice(indices)
            if not in_S[x] and bad_count[x] == 0:
                add(x)
                if current_score > best_score:
                    best_score = current_score
                    best_set = in_S[:]
                    
        elif r < 0.7:
            x = random.choice(indices)
            if in_S[x]:
                remove(x)
                delta = -weights[x]
                if random.random() < math.exp(delta / T):
                    if current_score > best_score:
                        best_score = current_score
                        best_set = in_S[:]
                else:
                    add(x)
                    
        else:
            x = random.choice(indices)
            y = random.choice(indices)
            if in_S[x] and not in_S[y]:
                K = 0
                for y_prime, z in incidents[x]:
                    if y_prime == y and in_S[z]:
                        K += 1
                        
                if bad_count[y] - K == 0:
                    remove(x)
                    add(y)
                    delta = weights[y] - weights[x]
                    if delta > 0 or random.random() < math.exp(delta / T):
                        if current_score > best_score:
                            best_score = current_score
                            best_set = in_S[:]
                    else:
                        remove(y)
                        add(x)
                        
        T *= alpha
        if T < min_T:
            T = min_T
            
    # Local Search Polish
    # Repeat until no improvement
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try all adds
        for i in range(1, N + 1):
            if not in_S[i] and bad_count[i] == 0:
                add(i)
                if current_score > best_score:
                    best_score = current_score
                    best_set = in_S[:]
                    improved = True
                else:
                    remove(i)
        # Try all removes
        for i in range(1, N + 1):
            if in_S[i]:
                remove(i)
                if current_score > best_score:
                    best_score = current_score
                    best_set = in_S[:]
                    improved = True
                else:
                    add(i)
        # Try all swaps
        for i in range(1, N + 1):
            if not in_S[i]:
                for j in range(1, N + 1):
                    if in_S[j]:
                        # Check swap j -> i
                        K = 0
                        for y_prime, z in incidents[j]:
                            if y_prime == i and in_S[z]:
                                K += 1
                        if bad_count[i] - K == 0:
                            remove(j)
                            add(i)
                            if current_score > best_score:
                                best_score = current_score
                                best_set = in_S[:]
                                improved = True
                            else:
                                remove(i)
                                add(j)

    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
