import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    N = 512
    is_prime = [True] * N
    is_prime[0] = False
    if N > 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * N
    weights = [0.0] * N
    
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        m = 0
        temp_a = a
        for p in primes:
            if p * p > temp_a:
                break
            if temp_a % p == 0:
                m |= (1 << prime_to_idx[p])
                while temp_a % p == 0:
                    temp_a //= p
        if temp_a > 1:
            m |= (1 << prime_to_idx[temp_a])
        masks[a] = m

    def can_add(a, mask):
        return (masks[a] & mask) == 0

    def local_search(initial_sol):
        current_sol = list(initial_sol)
        current_mask = 0
        for a in current_sol:
            current_mask |= masks[a]
        current_score = sum(weights[a] for a in current_sol)
        
        while time.time() < deadline - 2.0:
            improved = False
            sol_set = set(current_sol)
            
            # 1. Add
            best_add = -1
            best_w = 0.0
            for a in range(1, N):
                if a not in sol_set and can_add(a, current_mask):
                    if weights[a] > best_w:
                        best_w = weights[a]
                        best_add = a
            if best_add != -1:
                current_sol.append(best_add)
                current_mask |= masks[best_add]
                current_score += best_w
                improved = True
                continue

            # 2. 1-1 Swap
            best_i = -1
            best_j = -1
            best_gain = 0.0
            for i in current_sol:
                mask_wo_i = current_mask ^ masks[i]
                for j in range(1, N):
                    if j not in sol_set and can_add(j, mask_wo_i):
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_i = i
                            best_j = j
            if best_i != -1:
                current_sol.remove(best_i)
                current_sol.append(best_j)
                current_mask = (current_mask ^ masks[best_i]) | masks[best_j]
                current_score += best_gain
                improved = True
                continue

            # 3. 2-2 Swap (Remove i, k; Add j, l)
            # This allows moving from {A, B} to {C, D}
            # Complexity: O(S^2 * N^2) is too high.
            # Simplified: Try to replace two items with two others.
            # We can approximate by trying to swap i with j, then k with l.
            # But 1-1 already does that.
            # True 2-2 is needed for cases where individual swaps are bad but combined are good.
            # Given time, we'll skip explicit 2-2 and rely on restarts.
            
            if not improved:
                break
                
        return current_sol, current_score

    # Initialization
    best_sol = []
    best_score = 0.0
    
    # Greedy Init
    sol_greedy = []
    mask_greedy = 0
    for a in sorted(range(1, N), key=lambda x: -weights[x]):
        if can_add(a, mask_greedy):
            sol_greedy.append(a)
            mask_greedy |= masks[a]
            
    sol_res, score_res = local_search(sol_greedy)
    if score_res > best_score:
        best_score = score_res
        best_sol = sol_res

    # Randomized Restarts
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        # Random valid set
        sol_rand = []
        mask_rand = 0
        # Shuffle
        candidates = list(range(1, N))
        random.shuffle(candidates)
        for a in candidates:
            if random.random() < 0.2:
                if can_add(a, mask_rand):
                    sol_rand.append(a)
                    mask_rand |= masks[a]
        
        sol_res, score_res = local_search(sol_rand)
        if score_res > best_score:
            best_score = score_res
            best_sol = sol_res

    return best_sol
