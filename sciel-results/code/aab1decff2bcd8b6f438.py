import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    # Use 55s to be safe, evaluator kills at 90s but task says 60s limit usually.
    # "Keep an EXPLICIT deadline guard... safety margin before the limit."
    # Task says "Each evaluation must finish within 60 s".
    # So limit is 60s. I will use 58s.
    time_limit = 58.0
    
    # Precompute bit values
    bit_val = [1 << i for i in range(n)]
    
    # Precompute forbidden 4-AP masks
    # forbidden_masks[v] contains masks M such that if (S & M) == M, v cannot be added
    forbidden_masks = [[] for _ in range(n + 1)]
    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d, v+3d -> need {v+d, v+2d, v+3d}
        for d in range(1, n // 3 + 1):
            if v + 3 * d <= n:
                m = bit_val[v + d - 1] | bit_val[v + 2 * d - 1] | bit_val[v + 3 * d - 1]
                masks.append(m)
        # v is 2nd: v-d, v, v+d, v+2d -> need {v-d, v+d, v+2d}
        for d in range(1, n // 3 + 1):
            if v - d >= 1 and v + 2 * d <= n:
                m = bit_val[v - d - 1] | bit_val[v + d - 1] | bit_val[v + 2 * d - 1]
                masks.append(m)
        # v is 3rd: v-2d, v-d, v, v+d -> need {v-2d, v-d, v+d}
        for d in range(1, n // 3 + 1):
            if v - 2 * d >= 1 and v + d <= n:
                m = bit_val[v - 2 * d - 1] | bit_val[v - d - 1] | bit_val[v + d - 1]
                masks.append(m)
        # v is 4th: v-3d, v-2d, v-d, v -> need {v-3d, v-2d, v-d}
        for d in range(1, n // 3 + 1):
            if v - 3 * d >= 1:
                m = bit_val[v - 3 * d - 1] | bit_val[v - 2 * d - 1] | bit_val[v - d - 1]
                masks.append(m)
        forbidden_masks[v] = masks
        
    # Precompute 3-AP "witness" masks for each element
    ap3_masks = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        masks = []
        # u is 1st: u, u+d, u+2d -> need {u+d, u+2d}
        for d in range(1, (n - u) // 2 + 1):
            masks.append(bit_val[u + d - 1] | bit_val[u + 2 * d - 1])
        # u is 2nd: u-d, u, u+d -> need {u-d, u+d}
        for d in range(1, min(u - 1, n - u) + 1):
            masks.append(bit_val[u - d - 1] | bit_val[u + d - 1])
        # u is 3rd: u-2d, u-d, u -> need {u-2d, u-d}
        for d in range(1, (u - 1) // 2 + 1):
            masks.append(bit_val[u - 2 * d - 1] | bit_val[u - d - 1])
        ap3_masks[u] = masks
        
    def can_add(S_mask, v):
        for m in forbidden_masks[v]:
            if (S_mask & m) == m:
                return False
        return True

    def count_3aps(S_mask):
        S_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
        count = 0
        for a in S_set:
            for d in range(1, (n - a) // 2 + 1):
                if (a + d in S_set) and (a + 2 * d in S_set):
                    count += 1
        return count

    def calc_delta(S_mask, u, v):
        S_mask_without_u = S_mask & ~bit_val[u - 1]
        
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
                
        gained = 0
        for m in ap3_masks[v]:
            if (S_mask_without_u & m) == m:
                gained += 1
                
        return gained - lost

    # --- Simulated Annealing Strategy ---
    
    # Initial State: Greedy construction to get a decent start
    # This is better than random for SA
    S_mask = 0
    S_set = set()
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    for x in candidates:
        if can_add(S_mask, x):
            S_mask |= bit_val[x - 1]
            S_set.add(x)
            
    current_count = count_3aps(S_mask)
    best_count = current_count
    best_mask = S_mask
    best_set = S_set.copy()
    
    # Annealing Parameters
    # Start with high temperature, cool down
    # T starts at 10.0, decays to 0.01
    T = 10.0
    cooling_rate = 0.9999
    # We run until time limit
    
    # Precompute list of elements for faster access?
    # S_list and N_list need updating.
    # To avoid O(n) update every step, we can just iterate over all pairs (u,v) 
    # where u in [1..n] and v in [1..n], but check membership.
    # Or maintain S_set and N_set.
    
    # To speed up the inner loop, we can pre-compute the "neighbors" for each u?
    # No, delta depends on S.
    
    # Optimization: 
    # Instead of iterating all u in S and v in N, 
    # we can pick a random u in S and a random v in N.
    # This is standard SA.
    
    # Let's estimate speed.
    # One step: 
    # 1. Pick u, v. O(1) if we have lists.
    # 2. Check validity of v in S\{u}. O(K_v) ~ 50 checks.
    # 3. Calc delta. O(K_u + K_v) ~ 100 checks.
    # Total ~ 150 bitwise ops.
    # Python can do ~100k-1M simple ops/sec? No, Python is slower.
    # Maybe 50k steps/sec.
    # 58s * 50k = 2.9M steps. This is a lot.
    # We can afford to be less aggressive in cooling.
    
    # However, maintaining S_set and N_set efficiently is key.
    # Removing from set is O(1). Adding is O(1).
    
    # Let's implement the SA loop.
    
    while time.time() - start_time < time_limit:
        # Get current members
        # To avoid O(n) list creation every step, we can maintain the sets.
        # But picking random from set is O(1).
        
        # If set is empty, restart? No, we start greedy.
        if not S_set:
            break
            
        u = random.choice(list(S_set))
        
        # Find a valid v to swap to?
        # If we just pick random v, it might be invalid.
        # We should pick v from N_set.
        if not S_set:
             break
             
        # Efficiently pick v from N_set
        # N_set is [1..n] \ S_set
        # We can maintain N_set as a list? No, removal is slow.
        # We can just pick a random v in 1..n and check if it's in S_set?
        # If it is in S_set, it's not a swap (u->u), skip.
        # If it's not in S_set, it's a candidate.
        
        v = random.randint(1, n)
        if v == u:
            continue
        if v in S_set:
            # v is already in S, so this is not a swap u->v.
            # We need v NOT in S.
            # So if v in S_set, this move is invalid (not a swap).
            # We could retry, but let's just skip.
            continue
            
        # Check validity: Can v be added to S \ {u}?
        S_mask_without_u = S_mask & ~bit_val[u - 1]
        
        valid = True
        for m in forbidden_masks[v]:
            if (S_mask_without_u & m) == m:
                valid = False
                break
        
        if not valid:
            continue
            
        # Calculate delta
        delta = calc_delta(S_mask, u, v)
        
        # Acceptance criterion
        accept = False
        if delta > 0:
            accept = True
        else:
            # delta <= 0
            # P(accept) = exp(delta / T)
            # If T is small, exp(negative) is small.
            if T > 0:
                # Avoid overflow/underflow
                if delta / T > -100: 
                    prob = math.exp(delta / T)
                else:
                    prob = 0.0
                accept = random.random() < prob
            
        if accept:
            # Perform swap
            S_mask = S_mask_without_u | bit_val[v - 1]
            S_set.remove(u)
            S_set.add(v)
            current_count += delta
            
            if current_count > best_count:
                best_count = current_count
                best_mask = S_mask
                best_set = S_set.copy()
                
        # Cool down
        # We want to cool slowly to explore.
        # If we cool too fast, we get stuck.
        # If we cool too slow, we don't converge.
        T *= cooling_rate
        
    # Final verification
    if best_mask == 0:
        return [1]
        
    best_list = sorted(best_set)
    return best_list

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
