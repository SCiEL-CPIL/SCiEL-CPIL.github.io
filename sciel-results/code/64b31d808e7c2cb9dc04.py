import numpy as np
import time
import random

def construct():
    n = 12
    m = 40
    N = 1 << n
    deadline = time.time() + 50.0

    # 1. Precompute Comparability Matrix
    all_masks = np.arange(N, dtype=np.int32)
    M = np.zeros((N, N), dtype=np.uint8)
    
    for i in range(N):
        # i subset j
        sub1 = (i & all_masks) == i
        # j subset i
        sub2 = (all_masks & i) == all_masks
        row = sub1 | sub2
        row[i] = 0
        M[i, :] = row.astype(np.uint8)

    best_score = -1
    best_family = None

    # 2. Greedy Chain Initialization Function
    def greedy_init():
        family = []
        used = np.zeros(N, dtype=bool)
        # Start with a chain
        chain = [0]
        cur = 0
        for k in range(n):
            cur |= (1 << k)
            chain.append(cur)
        family.extend(chain)
        for c in chain:
            used[c] = True
        
        count_vec = np.zeros(N, dtype=np.int32)
        for c in chain:
            count_vec += M[c, :]
            
        # Add remaining m - len(chain) sets
        while len(family) < m:
            # Find best candidate not in used
            # Mask used
            masked_cv = count_vec.copy()
            masked_cv[used] = -1
            best_cand = int(np.argmax(masked_cv))
            
            if masked_cv[best_cand] <= 0:
                # If no good candidate, pick random
                available = np.where(~used)[0]
                best_cand = int(available[random.randint(0, len(available)-1)])
            
            family.append(best_cand)
            used[best_cand] = True
            count_vec += M[best_cand, :]
            
        return family, count_vec

    # 3. Local Search Function
    def local_search(family, count_vec, used, deadline):
        score = int(np.sum(count_vec) // 2)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle indices
            indices = list(range(m))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                A = family[i]
                old_loss = count_vec[A]
                
                # Remove A
                count_vec -= M[A, :]
                used[A] = False
                
                # Find best B
                masked = count_vec.copy()
                masked[used] = -1
                B = int(np.argmax(masked))
                new_gain = count_vec[B]
                
                if new_gain > old_loss:
                    # Accept
                    family[i] = B
                    used[B] = True
                    count_vec += M[B, :]
                    score += (new_gain - old_loss)
                    improved = True
                else:
                    # Revert
                    used[A] = True
                    count_vec += M[A, :]
        return score, family

    # 4. Main Loop: Restarts
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if r == 0:
            family, count_vec = greedy_init()
        else:
            # Random init for diversity
            family = random.sample(range(N), m)
            used = np.zeros(N, dtype=bool)
            for x in family:
                used[x] = True
            count_vec = np.zeros(N, dtype=np.int32)
            for x in family:
                count_vec += M[x, :]
        
        score, family = local_search(family, count_vec, used, deadline)
        
        if score > best_score:
            best_score = score
            best_family = family[:]

    if best_family is None:
        best_family = list(range(m))

    # Convert to lists of ints
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
