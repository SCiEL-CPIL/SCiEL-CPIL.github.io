import time
import random

# Precomputation
def precompute_aps(n):
    aps = []
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= n:
                aps.append((a, b, c, e))
    return aps

APS = precompute_aps(512)
APS_CONTAINING = [[] for _ in range(513)]
for i, ap in enumerate(APS):
    for x in ap:
        APS_CONTAINING[x].append(i)

def get_base3_set(n):
    s = set()
    for x in range(1, n+1):
        # check base 3 digits
        y = x
        valid = True
        while y > 0:
            if y % 3 == 2:
                valid = False
                break
            y //= 3
        if valid:
            s.add(x)
    return s

def greedy_fill(initial_set, n=512):
    S = set(initial_set)
    ap_counts = [0] * len(APS)
    
    # Initialize counts
    for x in S:
        for ap_idx in APS_CONTAINING[x]:
            ap_counts[ap_idx] += 1
            
    candidates = [x for x in range(1, n+1) if x not in S]
    random.shuffle(candidates)
    
    for x in candidates:
        valid = True
        for ap_idx in APS_CONTAINING[x]:
            if ap_counts[ap_idx] >= 3:
                valid = False
                break
        if valid:
            S.add(x)
            for ap_idx in APS_CONTAINING[x]:
                ap_counts[ap_idx] += 1
                
    return S

def construct():
    start_time = time.time()
    best_S = get_base3_set(512)
    best_size = len(best_S)
    
    # Run local search
    while time.time() - start_time < 50:
        # Perturb
        current_S = set(best_S)
        # Remove some random elements to allow new ones
        # Try different removal rates
        removal_rate = random.choice([0.05, 0.1, 0.2])
        num_remove = int(len(current_S) * removal_rate)
        to_remove = random.sample(list(current_S), num_remove)
        current_S -= set(to_remove)
        
        new_S = greedy_fill(current_S)
        
        if len(new_S) > best_size:
            best_size = len(new_S)
            best_S = new_S
            
    return sorted(best_S)
