import random
import math

def construct():
    N = 256
    
    # Precompute 4-APs
    four_aps = []
    ap_indices = [[] for _ in range(N+1)]
    for d in range(1, N//3 + 1):
        for a in range(1, N - 3*d + 1):
            ap = (a, a+d, a+2*d, a+3*d)
            idx = len(four_aps)
            four_aps.append(ap)
            for x in ap:
                ap_indices[x].append(idx)
            
    # Precompute 3-AP neighbors
    # For each x, list of (other1, other2) such that {x, other1, other2} is a 3-AP
    three_ap_neighbors = [[] for _ in range(N+1)]
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            # (a, b, c)
            three_ap_neighbors[a].append((b, c))
            three_ap_neighbors[b].append((a, c))
            three_ap_neighbors[c].append((a, b))
            
    def count_3aps_involving(x, A, exclude=None):
        count = 0
        for u, v in three_ap_neighbors[x]:
            if exclude is not None:
                if u == exclude or v == exclude:
                    continue
            if u in A and v in A:
                count += 1
        return count

    # Initialize with greedy set
    A = set()
    for x in range(1, N+1):
        valid = True
        for d in range(1, (x-1)//3 + 1):
            if all(x-j*d in A for j in (1,2,3)):
                valid = False
                break
        if valid:
            A.add(x)
            
    # Calculate initial score
    score = 0
    for x in A:
        score += count_3aps_involving(x, A)
    score //= 3
    
    # Initialize 4-AP counts
    ap_counts = [0] * len(four_aps)
    for x in A:
        for idx in ap_indices[x]:
            ap_counts[idx] += 1
            
    # Simulated Annealing
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    best_score = score
    best_A = set(A)
    
    iterations = 0
    max_iters = 500000 # Adjust based on time
    
    while T > min_T and iterations < max_iters:
        x_list = list(A)
        if not x_list:
            break
        x = random.choice(x_list)
        y = random.randint(1, N)
        if y in A:
            iterations += 1
            continue
            
        # Check 4-AP validity for swap
        # Remove x
        for idx in ap_indices[x]:
            ap_counts[idx] -= 1
        # Add y
        valid = True
        for idx in ap_indices[y]:
            ap_counts[idx] += 1
            if ap_counts[idx] == 4:
                valid = False
                break
                
        if not valid:
            for idx in ap_indices[y]:
                ap_counts[idx] -= 1
            for idx in ap_indices[x]:
                ap_counts[idx] += 1
            T *= cooling
            iterations += 1
            continue
            
        # Calculate delta
        loss = count_3aps_involving(x, A)
        gain = count_3aps_involving(y, A, exclude=x)
        delta = gain - loss
        
        if delta > 0 or random.random() < math.exp(delta / T):
            A.remove(x)
            A.add(y)
            score += delta
            if score > best_score:
                best_score = score
                best_A = set(A)
        # else reject, A remains unchanged
        
        T *= cooling
        iterations += 1
        
    return sorted(best_A)
