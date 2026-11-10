import time
import random

N = 256

# Precompute 4-APs
aps_4 = []
for d in range(1, (N-1)//3 + 1):
    for a in range(1, N - 3*d + 1):
        p = (a, a+d, a+2*d, a+3*d)
        aps_4.append(p)

x_to_aps = {i: [] for i in range(1, N+1)}
for idx, p in enumerate(aps_4):
    for x in p:
        x_to_aps[x].append(idx)

def is_valid_add(A_set, x):
    for idx in x_to_aps[x]:
        p = aps_4[idx]
        others = [y for y in p if y != x]
        if all(y in A_set for y in others):
            return False
    return True

def count_3aps(A_set):
    count = 0
    for b in A_set:
        for d in range(1, N):
            a = b - d
            c = b + d
            if a < 1 or c > N:
                continue
            if a in A_set and c in A_set:
                count += 1
    return count

def delta_3aps(A_set, v):
    delta = 0
    # v is middle
    for d in range(1, N):
        a = v - d
        c = v + d
        if a >= 1 and c <= N:
            if a in A_set and c in A_set:
                delta += 1
    # v is start
    for d in range(1, N):
        a = v + d
        c = v + 2*d
        if a <= N and c <= N:
            if a in A_set and c in A_set:
                delta += 1
    # v is end
    for d in range(1, N):
        b = v - d
        a = v - 2*d
        if b >= 1 and a >= 1:
            if a in A_set and b in A_set:
                delta += 1
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # 1. Initialize with Seed Greedy
    A = set()
    for x in range(1, N+1):
        if is_valid_add(A, x):
            A.add(x)
    
    # 2. Make Maximal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N+1):
            if x not in A and is_valid_add(A, x):
                A.add(x)
                changed = True
    
    current_score = count_3aps(A)
    best_score = current_score
    best_A = set(A)
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Randomly pick u to remove
        u = random.choice(list(A))
        A_temp = A - {u}
        base_score = current_score - delta_3aps(A_temp, u)
        
        best_v = None
        best_new_score = base_score
        
        for v in range(1, N+1):
            if v in A:
                continue
            if not is_valid_add(A_temp, v):
                continue
            
            new_score = base_score + delta_3aps(A_temp, v)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_v = v
        
        if best_v is not None and best_new_score > current_score:
            A.remove(u)
            A.add(best_v)
            current_score = best_new_score
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_A = set(A)
                
            # Try to add more
            for x in range(1, N+1):
                if x not in A and is_valid_add(A, x):
                    A.add(x)
                    current_score += delta_3aps(A - {x}, x)
                    if current_score > best_score:
                        best_score = current_score
                        best_A = set(A)
                    break
        else:
            no_improve_count += 1
            if no_improve_count > 1000:
                # Restart with a new random valid set
                # Simple random greedy
                A = set()
                order = list(range(1, N+1))
                random.shuffle(order)
                for x in order:
                    if is_valid_add(A, x):
                        A.add(x)
                # Make maximal
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for x in range(1, N+1):
                        if x not in A and is_valid_add(A, x):
                            A.add(x)
                            changed = True
                current_score = count_3aps(A)
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A)

    return sorted(best_A)
