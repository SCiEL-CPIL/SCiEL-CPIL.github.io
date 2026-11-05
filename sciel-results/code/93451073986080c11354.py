"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def try_greedy_and_search(n, seed):
        rng = random.Random(seed)
        # Initialize colors for triples of n vertices
        # We will build the coloring incrementally or randomly and then fix.
        # Greedy vertex addition is hard to backtrack. 
        # Let's use Random Init + Fast Local Search with Bitsets.
        
        triples = list(combinations(range(n), 3))
        m = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # Random Init
        colors = [rng.randint(0, 1) for _ in range(m)]
        
        # Precompute violation structures
        # For Red K4: list of 4 triple indices for each K4
        red_k4s = []
        for quad in combinations(range(n), 4):
            red_k4s.append((
                triple_idx[(quad[0], quad[1], quad[2])],
                triple_idx[(quad[0], quad[1], quad[3])],
                triple_idx[(quad[0], quad[2], quad[3])],
                triple_idx[(quad[1], quad[2], quad[3])]
            ))
            
        # For Blue K5: list of 10 triple indices for each K5
        blue_k5s = []
        for quint in combinations(range(n), 5):
            k5_triples = [triple_idx[tri] for tri in combinations(quint, 3)]
            blue_k5s.append(k5_triples)
            
        # To speed up violation counting, we can maintain counts?
        # Or just count each time if it's fast enough.
        # m ~ 6000 for n=34. 
        # Red K4s ~ 46k. Blue K5s ~ 278k.
        # Checking all blue K5s is 278k * 10 checks = 2.7M ops.
        # In Python, 2.7M ops per iteration is too slow for 1000 iterations.
        
        # Optimization: Only check K5s that are "close" to all blue?
        # Or use a different approach: 
        # Flip a triple, update violations involving that triple.
        
        # Let's try a different heuristic: 
        # Minimize a score function: Sum of (red K4 count) + (blue K5 count).
        # But we need 0 violations.
        
        # Faster Local Search:
        # 1. Find all current violations.
        # 2. Pick a random violation.
        # 3. Flip a random triple in it.
        # 4. Recalculate violations only for structures affected by the flip?
        #    A flip affects K4s containing that triple and K5s containing that triple.
        
        # Let's precompute for each triple index, which K4s and K5s it belongs to.
        triple_to_k4s = [[] for _ in range(m)]
        for i, k4 in enumerate(red_k4s):
            for t in k4:
                triple_to_k4s[t].append(i)
                
        triple_to_k5s = [[] for _ in range(m)]
        for i, k5 in enumerate(blue_k5s):
            for t in k5:
                triple_to_k5s[t].append(i)
                
        # State: colors array.
        # Violation state: is_red_viol[i], is_blue_viol[i]
        is_red_viol = [False] * len(red_k4s)
        is_blue_viol = [False] * len(blue_k5s)
        
        # Initial count
        red_viol_count = 0
        for i, k4 in enumerate(red_k4s):
            if all(colors[t] == 0 for t in k4):
                is_red_viol[i] = True
                red_viol_count += 1
                
        blue_viol_count = 0
        for i, k5 in enumerate(blue_k5s):
            if all(colors[t] == 1 for t in k5):
                is_blue_viol[i] = True
                blue_viol_count += 1
                
        best_red = red_viol_count
        best_blue = blue_viol_count
        best_colors = colors[:]
        
        max_iters = 50000 # Try many iterations
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            if red_viol_count == 0 and blue_viol_count == 0:
                return colors
                
            # Pick a violation
            # Weighted random: if red_viol > 0, pick red, else blue
            # Or just pick from all violations
            if red_viol_count > 0 and (blue_viol_count == 0 or rng.random() < 0.5):
                # Pick random red violation
                while True:
                    vi = rng.randint(0, len(red_k4s) - 1)
                    if is_red_viol[vi]:
                        break
                k4 = red_k4s[vi]
                # Flip a random triple in k4
                flip_t = k4[rng.randint(0, 3)]
                old_c = colors[flip_t]
                new_c = 1 - old_c
                colors[flip_t] = new_c
                
                # Update violations for K4s and K5s involving flip_t
                # Update Red K4s
                for k4_idx in triple_to_k4s[flip_t]:
                    # Check if this K4 is now all red
                    # Optimization: count reds in k4
                    # If old_c was 0, it was red. Now it's blue. So red count decreased by 1.
                    # If old_c was 1, it was blue. Now it's red. So red count increased by 1.
                    # We don't track counts, so we just re-check the 4 triples.
                    k4_triples = red_k4s[k4_idx]
                    # Fast check
                    is_all_red = True
                    for t in k4_triples:
                        if colors[t] != 0:
                            is_all_red = False
                            break
                    if is_all_red:
                        if not is_red_viol[k4_idx]:
                            is_red_viol[k4_idx] = True
                            red_viol_count += 1
                    else:
                        if is_red_viol[k4_idx]:
                            is_red_viol[k4_idx] = False
                            red_viol_count -= 1
                            
                # Update Blue K5s
                for k5_idx in triple_to_k5s[flip_t]:
                    k5_triples = blue_k5s[k5_idx]
                    is_all_blue = True
                    for t in k5_triples:
                        if colors[t] != 1:
                            is_all_blue = False
                            break
                    if is_all_blue:
                        if not is_blue_viol[k5_idx]:
                            is_blue_viol[k5_idx] = True
                            blue_viol_count += 1
                    else:
                        if is_blue_viol[k5_idx]:
                            is_blue_viol[k5_idx] = False
                            blue_viol_count -= 1
                            
            else:
                # Pick random blue violation
                if blue_viol_count == 0:
                    # No blue violations, but red exists? 
                    # If red_viol > 0, we should have picked red.
                    # This case implies red_viol == 0 and blue_viol > 0
                    while True:
                        vi = rng.randint(0, len(blue_k5s) - 1)
                        if is_blue_viol[vi]:
                            break
                    k5 = blue_k5s[vi]
                    flip_t = k5[rng.randint(0, 9)]
                    old_c = colors[flip_t]
                    new_c = 1 - old_c
                    colors[flip_t] = new_c
                    
                    # Update Red K4s
                    for k4_idx in triple_to_k4s[flip_t]:
                        k4_triples = red_k4s[k4_idx]
                        is_all_red = True
                        for t in k4_triples:
                            if colors[t] != 0:
                                is_all_red = False
                                break
                        if is_all_red:
                            if not is_red_viol[k4_idx]:
                                is_red_viol[k4_idx] = True
                                red_viol_count += 1
                        else:
                            if is_red_viol[k4_idx]:
                                is_red_viol[k4_idx] = False
                                red_viol_count -= 1
                                
                    # Update Blue K5s
                    for k5_idx in triple_to_k5s[flip_t]:
                        k5_triples = blue_k5s[k5_idx]
                        is_all_blue = True
                        for t in k5_triples:
                            if colors[t] != 1:
                                is_all_blue = False
                                break
                        if is_all_blue:
                            if not is_blue_viol[k5_idx]:
                                is_blue_viol[k5_idx] = True
                                blue_viol_count += 1
                        else:
                            if is_blue_viol[k5_idx]:
                                is_blue_viol[k5_idx] = False
                                blue_viol_count -= 1
                                
            if red_viol_count < best_red or (red_viol_count == best_red and blue_viol_count < best_blue):
                best_red = red_viol_count
                best_blue = blue_viol_count
                best_colors = colors[:]
                
        if best_red == 0 and best_blue == 0:
            return best_colors
        return None

    # Try to find valid coloring for n=34, 33, etc.
    # Start from high n and go down? Or start low and go up?
    # Known bound is 34. Let's try 34 first with multiple seeds.
    # If 34 fails, try 33, etc.
    
    best_n = 0
    best_colors = None
    
    # Try n=34
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        res = try_greedy_and_search(34, seed)
        if res:
            best_n = 34
            best_colors = res
            break
            
    # If 34 failed, try 33
    if best_n < 33:
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            res = try_greedy_and_search(33, seed)
            if res:
                best_n = 33
                best_colors = res
                break
                
    # If 33 failed, try 32
    if best_n < 32:
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            res = try_greedy_and_search(32, seed)
            if res:
                best_n = 32
                best_colors = res
                break
                
    # Fallback to smaller n if all fail
    if best_n < 32:
        for n in range(31, 10, -1):
            if time.time() > deadline - 5:
                break
            res = try_greedy_and_search(n, 12345)
            if res:
                best_n = n
                best_colors = res
                break
                
    # Absolute fallback
    if best_n == 0:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
        
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
