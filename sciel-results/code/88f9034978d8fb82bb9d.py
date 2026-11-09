import random
import time
import math

def construct():
    n = 256
    
    # Precompute 3-AP and 4-AP neighbors
    pairs_3 = [[] for _ in range(n + 1)]
    triples_4 = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            pairs_3[a].append((b, c))
            pairs_3[b].append((a, c))
            pairs_3[c].append((a, b))
            
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            triples_4[a].append((b, c, e))
            triples_4[b].append((a, c, e))
            triples_4[c].append((a, b, e))
            triples_4[e].append((a, b, c))

    # Helper to count 3-APs for a set (used for initialization and final check)
    def count_aps_full(A_set):
        in_A = [False] * (n + 1)
        for x in A_set:
            in_A[x] = True
        c3 = 0
        for x in A_set:
            for y, z in pairs_3[x]:
                if in_A[y] and in_A[z]:
                    c3 += 1
        return c3 // 3

    # Initial Construction: Base-4 digit restriction (digits 0,1,2)
    def valid_digit(x):
        while x > 0:
            if x % 4 > 2:
                return False
            x //= 4
        return True
        
    A_set = set()
    for i in range(1, n + 1):
        if valid_digit(i - 1):
            A_set.add(i)
            
    in_A = [False] * (n + 1)
    for x in A_set:
        in_A[x] = True
        
    # Calculate initial score
    current_score = count_aps_full(A_set)
    best_score = current_score
    best_A = set(A_set)
    
    start_time = time.time()
    timeout = 55.0
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() - start_time < timeout and T > T_min:
        op = random.choice(['add', 'remove', 'swap'])
        
        if op == 'add':
            candidates = [x for x in range(1, n + 1) if not in_A[x]]
            if not candidates:
                continue
            x = random.choice(candidates)
            
            # Check 4-AP validity
            valid = True
            for y, z, w in triples_4[x]:
                if in_A[y] and in_A[z] and in_A[w]:
                    valid = False
                    break
            if not valid:
                continue
                
            # Calculate gain
            gain = 0
            for y, z in pairs_3[x]:
                if in_A[y] and in_A[z]:
                    gain += 1
                    
            delta = gain
            
            if delta > 0 or random.random() < math.exp(delta / T):
                in_A[x] = True
                A_set.add(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A_set)
                    
        elif op == 'remove':
            if not A_set:
                continue
            x = random.choice(list(A_set))
            
            # Calculate loss
            loss = 0
            for y, z in pairs_3[x]:
                if in_A[y] and in_A[z]:
                    loss += 1
                    
            delta = -loss
            
            if delta > 0 or random.random() < math.exp(delta / T):
                in_A[x] = False
                A_set.remove(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A_set)
                    
        elif op == 'swap':
            if not A_set:
                continue
            x = random.choice(list(A_set))
            candidates = [y for y in range(1, n + 1) if not in_A[y]]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Check 4-AP validity for y given x is removed
            # Temporarily mark x as removed for check
            in_A[x] = False
            
            valid = True
            for u, v, w in triples_4[y]:
                if in_A[u] and in_A[v] and in_A[w]:
                    valid = False
                    break
                    
            if not valid:
                in_A[x] = True # restore
                continue
                
            # Calculate change
            # Loss from removing x
            loss_x = 0
            for u, v in pairs_3[x]:
                if in_A[u] and in_A[v]: # in_A has x=False
                    loss_x += 1
                    
            # Gain from adding y
            gain_y = 0
            for u, v in pairs_3[y]:
                if in_A[u] and in_A[v]: # in_A has x=False
                    gain_y += 1
                    
            delta = gain_y - loss_x
            
            if delta > 0 or random.random() < math.exp(delta / T):
                # Apply swap
                # x is already False in in_A
                in_A[y] = True
                A_set.remove(x)
                A_set.add(y)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A_set)
            else:
                # Revert
                in_A[x] = True
                
        T *= cooling_rate

    return [i for i in range(1, n + 1) if i in best_A]
