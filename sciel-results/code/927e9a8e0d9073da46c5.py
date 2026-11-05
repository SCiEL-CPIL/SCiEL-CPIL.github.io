import time
import math
import random

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    candidates = list(range(1, N + 1))
    
    def try_greedy(order):
        S = []
        L = 1
        sums = {0}
        
        for x in order:
            L_new = math.lcm(L, x)
            w_new = L_new // x
            scale = L_new // L
            
            # Check for collision
            # We need to verify that {s * scale + w_new | s in sums} is disjoint from {s * scale | s in sums}
            # This is equivalent to checking if (s1 * scale + w_new) == (s2 * scale) for some s1, s2 in sums
            # s1 * scale - s2 * scale = -w_new
            # (s1 - s2) * scale = -w_new
            # s1 - s2 = -w_new / scale
            # Note that w_new / scale = (L_new / x) / (L_new / L) = L / x
            # So we check if L/x is an integer difference between two sums in `sums`?
            # Wait, s1 and s2 are integers from the previous step (scaled by L).
            # The new sums are scaled by L_new.
            # Let's stick to the set intersection method as it is less prone to algebraic errors and fast enough.
            
            # Optimization: If scale == 1, we just check if (s + w_new) in sums
            # If scale != 1, we have to create scaled_sums.
            
            # To avoid creating a new set if possible:
            # We can check:
            # For each s in sums, let s_new = s * scale.
            # We need to know if s_new + w_new is in {t * scale for t in sums}.
            # This is equivalent to: (s_new + w_new) % scale == 0 AND (s_new + w_new) // scale in sums.
            
            collision = False
            if scale == 1:
                for s in sums:
                    if (s + w_new) in sums:
                        collision = True
                        break
            else:
                # Pre-compute sums for faster lookup? No, sums is already a set.
                # But we need to check if (s*scale + w_new) is a multiple of scale and the quotient is in sums.
                # (s*scale + w_new) % scale == w_new % scale
                # So we just need w_new % scale == 0?
                # w_new = L_new / x. scale = L_new / L.
                # w_new / scale = L / x.
                # If L is not divisible by x, then w_new is not divisible by scale?
                # Actually, L_new is a multiple of both L and x.
                # L_new = k * L = m * x.
                # w_new = L_new / x = k * L / x.
                # scale = k.
                # w_new / scale = L / x.
                # If x does not divide L, L/x is not an integer.
                # Then w_new is not an integer multiple of scale?
                # w_new = (L/x) * scale.
                # If L/x is not integer, then w_new / scale is not integer.
                # Then s*scale + w_new cannot be equal to t*scale (integer) because the remainder modulo scale would be w_new % scale != 0.
                # So if L % x != 0, there is NO collision?
                # Let's verify.
                # If L % x != 0, then 1/x introduces a new "denominator" factor.
                # The sums of reciprocals of S have denominator dividing L.
                # 1/x has denominator x.
                # If gcd(L, x) < x, then the denominator of the new sums will be lcm(L, x).
                # The old sums, when expressed with denominator L_new, are multiples of scale = L_new/L.
                # The new sums are old_sums * scale + w_new.
                # w_new = L_new / x.
                # Is it possible that s1 * scale + w_new = s2 * scale?
                # s1 * scale - s2 * scale = -w_new
                # (s1 - s2) * scale = -w_new
                # s1 - s2 = -w_new / scale = -L/x
                # Since s1, s2 are integers, L/x must be an integer.
                # So if L is not divisible by x, L/x is not an integer, so no solution.
                # Thus, if L % x != 0, the set is ALWAYS dissociated?
                
                # Wait, this implies that if we pick numbers such that each new number does not divide the LCM of the previous ones, we can always add it?
                # This would mean we can add ALL numbers 1..22?
                # Let's check.
                # Start with 1. L=1.
                # Add 2. L=1. 1%2 != 0. So always add?
                # S={1,2}. L=2.
                # Add 3. L=2. 2%3 != 0. So always add?
                # S={1,2,3}. L=6.
                # Add 4. L=6. 6%4 != 0. So always add?
                # S={1,2,3,4}. L=12.
                # Add 5. L=12. 12%5 != 0. Add.
                # ...
                # Add 6. L=lcm(1..5)=60. 60%6==0.
                # So we need to check collision for 6.
                
                # This is a huge insight!
                # If x does not divide L, it is ALWAYS safe to add.
                # We only need to check for collision if x divides L.
                
                if L % x != 0:
                    collision = False
                else:
                    # x divides L, so scale = 1?
                    # L_new = L. scale = 1.
                    # w_new = L / x.
                    # Check if (s + w_new) in sums.
                    for s in sums:
                        if (s + w_new) in sums:
                            collision = True
                            break
        
            if not collision:
                S.append(x)
                L = L_new
                if scale == 1:
                    # sums = sums union (sums + w_new)
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + w_new)
                    sums = new_sums
                else:
                    # sums = {s * scale for s in sums} union {s * scale + w_new for s in sums}
                    scaled_sums = {s * scale for s in sums}
                    new_sums = set(scaled_sums)
                    for s in scaled_sums:
                        new_sums.add(s + w_new)
                    sums = new_sums
                
                if time.time() - start_time > timeout - 2:
                    break
        return S

    best_set = []
    
    # 1. Try descending order
    order_desc = sorted(candidates, reverse=True)
    s1 = try_greedy(order_desc)
    if len(s1) > len(best_set):
        best_set = s1
        
    # 2. Try ascending order
    order_asc = sorted(candidates)
    s2 = try_greedy(order_asc)
    if len(s2) > len(best_set):
        best_set = s2
        
    # 3. Try random orders
    max_restarts = 200
    for _ in range(max_restarts):
        if time.time() - start_time > timeout - 5:
            break
        random.shuffle(candidates)
        s_r = try_greedy(candidates)
        if len(s_r) > len(best_set):
            best_set = s_r
            
    if not best_set:
        best_set = [1]
        
    return best_set
