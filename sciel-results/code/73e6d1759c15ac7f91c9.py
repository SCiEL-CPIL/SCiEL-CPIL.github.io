import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    # Strategy: Deterministic algebraic construction over Z_n
    # We try n=34 first, but the algebraic structure is naturally defined on Z_n.
    # For n=34, we can use a cyclic construction on Z_34 or embed in Z_35.
    # However, the task requires vertices 0..n-1.
    # A cyclic coloring on Z_n is defined by f(a,b,c) = g((b-a) mod n, (c-b) mod n) or similar.
    # Let's try a simple quadratic residue based coloring on Z_35 and then map to 34?
    # No, we must color triples of 0..n-1.
    # Let's try n=35 first? The limit is n<=40.
    # If we can find a valid coloring for n=35, that's better than 34.
    # Let's try to construct for n=35 using Z_35 symmetry.
    
    # Quadratic residues mod 35:
    # QR(35) = {x^2 mod 35 | x in 0..34}
    # 0^2=0, 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25, 6^2=36=1, 7^2=49=14, 8^2=64=29, 9^2=81=11
    # 10^2=100=30, 11^2=121=16, 12^2=144=4, 13^2=169=29, 14^2=196=1, 15^2=225=10, 16^2=256=31, 17^2=289=14
    # 18^2=324=29, 19^2=361=11, 20^2=400=16, 21^2=441=1, 22^2=484=4, 23^2=529=9, 24^2=576=14, 25^2=625=20
    # 26^2=676=31, 27^2=729=10, 28^2=784=16, 29^2=841=29, 30^2=900=14, 31^2=961=1, 32^2=1024=9, 33^2=1089=4
    # 34^2=1156=1
    # QR = {0, 1, 4, 9, 10, 11, 14, 16, 20, 25, 29, 30, 31}
    
    # Let's try a cyclic coloring for n=35.
    # Color triple (a,b,c) with a<b<c.
    # Let d1 = (b-a) mod 35, d2 = (c-b) mod 35.
    # Or simply use the difference (c-a) mod 35.
    # If (c-a) mod 35 is in QR, color Red (0), else Blue (1).
    # Or vice versa.
    
    # Let's test this idea quickly for n=35.
    n_target = 35
    triples = list(combinations(range(n_target), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = list(combinations(range(n_target), 4))
    k5s = list(combinations(range(n_target), 5))
    
    # Precompute QR for 35
    qr_35 = set((x*x) % 35 for x in range(35))
    
    # Generate candidate colorings based on QR of (c-a) mod 35
    # We can also shift the QR set or use different moduli.
    # Let's try a few variants.
    
    best_result = None
    best_n = 0
    
    # Variant 1: Color Red if (c-a) mod n in QR
    def gen_coloring_variant1(n):
        qr = set((x*x) % n for x in range(n))
        colors = []
        for a, b, c in combinations(range(n), 3):
            d = (c - a) % n
            if d in qr:
                colors.append(0)
            else:
                colors.append(1)
        return colors
    
    # Variant 2: Color Red if (c-a) mod n NOT in QR
    def gen_coloring_variant2(n):
        qr = set((x*x) % n for x in range(n))
        colors = []
        for a, b, c in combinations(range(n), 3):
            d = (c - a) % n
            if d in qr:
                colors.append(1)
            else:
                colors.append(0)
        return colors
        
    # Variant 3: Use (b-a) and (c-b)
    def gen_coloring_variant3(n):
        qr = set((x*x) % n for x in range(n))
        colors = []
        for a, b, c in combinations(range(n), 3):
            d1 = (b - a) % n
            d2 = (c - b) % n
            # If both are QR, Red?
            if d1 in qr and d2 in qr:
                colors.append(0)
            else:
                colors.append(1)
        return colors
        
    # Variant 4: Sum of squares?
    def gen_coloring_variant4(n):
        qr = set((x*x) % n for x in range(n))
        colors = []
        for a, b, c in combinations(range(n), 3):
            d1 = (b - a) % n
            d2 = (c - b) % n
            d3 = (a - c) % n # which is -(d1+d2)
            cnt_qr = (1 if d1 in qr else 0) + (1 if d2 in qr else 0) + (1 if d3 in qr else 0)
            if cnt_qr >= 2:
                colors.append(0)
            else:
                colors.append(1)
        return colors
        
    # We will test these for n=35, 34, 33...
    # But we need to verify validity.
    
    def is_valid(n, colors):
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check K4 red
        for k4 in combinations(range(n), 4):
            t1 = (k4[0], k4[1], k4[2])
            t2 = (k4[0], k4[1], k4[3])
            t3 = (k4[0], k4[2], k4[3])
            t4 = (k4[1], k4[2], k4[3])
            if colors[triple_to_idx[t1]] == 0 and \
               colors[triple_to_idx[t2]] == 0 and \
               colors[triple_to_idx[t3]] == 0 and \
               colors[triple_to_idx[t4]] == 0:
                return False
                
        # Check K5 blue
        for k5 in combinations(range(n), 5):
            v = list(k5)
            all_blue = True
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (v[i], v[j], v[k])
                        if colors[triple_to_idx[t]] != 1:
                            all_blue = False
                            break
                    if not all_blue: break
                if not all_blue: break
            if all_blue:
                return False
        return True
        
    # Try n=35 with variants
    for n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 10:
            break
            
        variants = [
            gen_coloring_variant1,
            gen_coloring_variant2,
            gen_coloring_variant3,
            gen_coloring_variant4
        ]
        
        for gen_func in variants:
            if time.time() > deadline - 5:
                break
            try:
                colors = gen_func(n)
                if is_valid(n, colors):
                    return {'n': n, 'colors': colors}
            except Exception:
                pass

    # If algebraic constructions fail, fall back to local search for n=34
    # This is a hybrid: try fast deterministic first, then slow stochastic.
    
    target_n = 34
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = list(combinations(range(target_n), 4))
    k5s = list(combinations(range(target_n), 5))
    
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for k4_idx, k4 in enumerate(k4s):
        t1 = (k4[0], k4[1], k4[2])
        t2 = (k4[0], k4[1], k4[3])
        t3 = (k4[0], k4[2], k4[3])
        t4 = (k4[1], k4[2], k4[3])
        triple_to_k4s[triple_to_idx[t1]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t2]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t3]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t4]].append(k4_idx)
        
    for k5_idx, k5 in enumerate(k5s):
        v = list(k5)
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    t = (v[i], v[j], v[k])
                    triple_to_k5s[triple_to_idx[t]].append(k5_idx)

    # Local Search on n=34
    best_violations = float('inf')
    best_colors = None
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        k4_red_counts = [0] * len(k4s)
        k5_blue_counts = [0] * len(k5s)
        
        for k4_idx, k4 in enumerate(k4s):
            cnt = 0
            t1 = (k4[0], k4[1], k4[2])
            t2 = (k4[0], k4[1], k4[3])
            t3 = (k4[0], k4[2], k4[3])
            t4 = (k4[1], k4[2], k4[3])
            if colors[triple_to_idx[t1]] == 0: cnt += 1
            if colors[triple_to_idx[t2]] == 0: cnt += 1
            if colors[triple_to_idx[t3]] == 0: cnt += 1
            if colors[triple_to_idx[t4]] == 0: cnt += 1
            k4_red_counts[k4_idx] = cnt
            
        for k5_idx, k5 in enumerate(k5s):
            cnt = 0
            v = list(k5)
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (v[i], v[j], v[k])
                        if colors[triple_to_idx[t]] == 1: cnt += 1
            k5_blue_counts[k5_idx] = cnt

        v4 = sum(1 for c in k4_red_counts if c == 4)
        v5 = sum(1 for c in k5_blue_counts if c == 10)
        total_violations = v4 + v5
        
        if total_violations == 0:
            return {'n': target_n, 'colors': colors}
            
        if total_violations < best_violations:
            best_violations = total_violations
            best_colors = colors.copy()
            
        no_improve_steps = 0
        max_steps = 10000
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta = 0
            
            if old_color == 0: # Red -> Blue
                for k4_idx in triple_to_k4s[i]:
                    old_c = k4_red_counts[k4_idx]
                    if old_c == 4:
                        delta -= 1
                for k5_idx in triple_to_k5s[i]:
                    old_c = k5_blue_counts[k5_idx]
                    if old_c == 9:
                        delta += 1
            else: # Blue -> Red
                for k4_idx in triple_to_k4s[i]:
                    old_c = k4_red_counts[k4_idx]
                    if old_c == 3:
                        delta += 1
                for k5_idx in triple_to_k5s[i]:
                    old_c = k5_blue_counts[k5_idx]
                    if old_c == 10:
                        delta -= 1

            if delta < 0:
                colors[i] = new_color
                if new_color == 1:
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_counts[k4_idx] -= 1
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_counts[k5_idx] += 1
                else:
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_counts[k4_idx] += 1
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_counts[k5_idx] -= 1
                        
                total_violations += delta
                no_improve_steps = 0
                
                if total_violations == 0:
                    return {'n': target_n, 'colors': colors}
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    
            elif delta == 0:
                if random.random() < 0.1:
                    colors[i] = new_color
                    if new_color == 1:
                        for k4_idx in triple_to_k4s[i]:
                            k4_red_counts[k4_idx] -= 1
                        for k5_idx in triple_to_k5s[i]:
                            k5_blue_counts[k5_idx] += 1
                    else:
                        for k4_idx in triple_to_k4s[i]:
                            k4_red_counts[k4_idx] += 1
                        for k5_idx in triple_to_k5s[i]:
                            k5_blue_counts[k5_idx] -= 1
                    no_improve_steps += 1
            else:
                pass
                
            if total_violations < best_violations:
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            if no_improve_steps > 3000:
                break
                
    if best_violations == 0:
        return {'n': target_n, 'colors': best_colors}
        
    # Fallback
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from itertools import combinations
    
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    # Strategy: Deterministic algebraic construction over Z_n
    # We try n=35 first, but the algebraic structure is naturally defined on Z_n.
    # For n=34, we can use a cyclic construction on Z_34 or embed in Z_35.
    # However, the task requires vertices 0..n-1.
    # Let's try a simple quadratic residue based coloring on Z_35 and then map to 34?
    # No, we must color triples of 0..n-1.
    # Let's try n=35 first? The limit is n<=40.
    # If we can find a valid coloring for n=35, that's better than 34.
    # Let's try to construct for n=35 using Z_35 symmetry.
    
    # Quadratic residues mod 35:
    # QR(35) = {x^2 mod 35 | x in 0..34}
    
    # Let's try a cyclic coloring for n=35.
    # Color triple (a,b,c) with a<b<c.
    # Let d1 = (b-a) mod 35, d2 = (c-b) mod 35.
    # Or simply use the difference (c-a) mod 35.
    # If (c-a) mod 35 is in QR, color Red (0), else Blue (1).
    # Or vice versa.
    
    # Let's test this idea quickly for n=35.
    n_target = 35
    triples = list(combinations(range(n_target), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = list(combinations(range(n_target), 4))
    k5s = list(combinations(range(n_target), 5))
    
    # Precompute QR for 35
    qr_35 = set((x*x) % 35 for x in range(35))
    
    # Generate candidate colorings based on QR of (c-a) mod 35
    # We can also shift the QR set or use different moduli.
    # Let's try a few variants.
    
    best_result = None
    best_n = 0
    
    # Variant 1: Color Red if (c-a) mod n in QR
    def gen_coloring_variant1(n):
        qr = set((x*x) % n for x in range(n))
        colors = []
        for a, b, c in combinations(range(n), 3):
            d = (c - a) % n
            if d in qr:
                colors.append(0)
            else:
                colors.append(1)
        return colors
    
    # Variant 2: Color Red if (c-a) mod n NOT in QR
    def gen_coloring_variant2(n):
        qr = set((x*x) % n for x in range(n))
        colors = []
        for a, b, c in combinations(range(n), 3):
            d = (c - a) % n
            if d in qr:
                colors.append(1)
            else:
                colors.append(0)
        return colors
        
    # Variant 3: Use (b-a) and (c-b)
    def gen_coloring_variant3(n):
        qr = set((x*x) % n for x in range(n))
        colors = []
        for a, b, c in combinations(range(n), 3):
            d1 = (b - a) % n
            d2 = (c - b) % n
            # If both are QR, Red?
            if d1 in qr and d2 in qr:
                colors.append(0)
            else:
                colors.append(1)
        return colors
        
    # Variant 4: Sum of squares?
    def gen_coloring_variant4(n):
        qr = set((x*x) % n for x in range(n))
        colors = []
        for a, b, c in combinations(range(n), 3):
            d1 = (b - a) % n
            d2 = (c - b) % n
            d3 = (a - c) % n # which is -(d1+d2)
            cnt_qr = (1 if d1 in qr else 0) + (1 if d2 in qr else 0) + (1 if d3 in qr else 0)
            if cnt_qr >= 2:
                colors.append(0)
            else:
                colors.append(1)
        return colors
        
    # We will test these for n=35, 34, 33...
    # But we need to verify validity.
    
    def is_valid(n, colors):
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check K4 red
        for k4 in combinations(range(n), 4):
            t1 = (k4[0], k4[1], k4[2])
            t2 = (k4[0], k4[1], k4[3])
            t3 = (k4[0], k4[2], k4[3])
            t4 = (k4[1], k4[2], k4[3])
            if colors[triple_to_idx[t1]] == 0 and \
               colors[triple_to_idx[t2]] == 0 and \
               colors[triple_to_idx[t3]] == 0 and \
               colors[triple_to_idx[t4]] == 0:
                return False
                
        # Check K5 blue
        for k5 in combinations(range(n), 5):
            v = list(k5)
            all_blue = True
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (v[i], v[j], v[k])
                        if colors[triple_to_idx[t]] != 1:
                            all_blue = False
                            break
                    if not all_blue: break
                if not all_blue: break
            if all_blue:
                return False
        return True
        
    # Try n=35 with variants
    for n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 10:
            break
            
        variants = [
            gen_coloring_variant1,
            gen_coloring_variant2,
            gen_coloring_variant3,
            gen_coloring_variant4
        ]
        
        for gen_func in variants:
            if time.time() > deadline - 5:
                break
            try:
                colors = gen_func(n)
                if is_valid(n, colors):
                    return {'n': n, 'colors': colors}
            except Exception:
                pass

    # If algebraic constructions fail, fall back to local search for n=34
    # This is a hybrid: try fast deterministic first, then slow stochastic.
    
    target_n = 34
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = list(combinations(range(target_n), 4))
    k5s = list(combinations(range(target_n), 5))
    
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for k4_idx, k4 in enumerate(k4s):
        t1 = (k4[0], k4[1], k4[2])
        t2 = (k4[0], k4[1], k4[3])
        t3 = (k4[0], k4[2], k4[3])
        t4 = (k4[1], k4[2], k4[3])
        triple_to_k4s[triple_to_idx[t1]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t2]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t3]].append(k4_idx)
        triple_to_k4s[triple_to_idx[t4]].append(k4_idx)
        
    for k5_idx, k5 in enumerate(k5s):
        v = list(k5)
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    t = (v[i], v[j], v[k])
                    triple_to_k5s[triple_to_idx[t]].append(k5_idx)

    # Local Search on n=34
    best_violations = float('inf')
    best_colors = None
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        k4_red_counts = [0] * len(k4s)
        k5_blue_counts = [0] * len(k5s)
        
        for k4_idx, k4 in enumerate(k4s):
            cnt = 0
            t1 = (k4[0], k4[1], k4[2])
            t2 = (k4[0], k4[1], k4[3])
            t3 = (k4[0], k4[2], k4[3])
            t4 = (k4[1], k4[2], k4[3])
            if colors[triple_to_idx[t1]] == 0: cnt += 1
            if colors[triple_to_idx[t2]] == 0: cnt += 1
            if colors[triple_to_idx[t3]] == 0: cnt += 1
            if colors[triple_to_idx[t4]] == 0: cnt += 1
            k4_red_counts[k4_idx] = cnt
            
        for k5_idx, k5 in enumerate(k5s):
            cnt = 0
            v = list(k5)
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (v[i], v[j], v[k])
                        if colors[triple_to_idx[t]] == 1: cnt += 1
            k5_blue_counts[k5_idx] = cnt

        v4 = sum(1 for c in k4_red_counts if c == 4)
        v5 = sum(1 for c in k5_blue_counts if c == 10)
        total_violations = v4 + v5
        
        if total_violations == 0:
            return {'n': target_n, 'colors': colors}
            
        if total_violations < best_violations:
            best_violations = total_violations
            best_colors = colors.copy()
            
        no_improve_steps = 0
        max_steps = 10000
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta = 0
            
            if old_color == 0: # Red -> Blue
                for k4_idx in triple_to_k4s[i]:
                    old_c = k4_red_counts[k4_idx]
                    if old_c == 4:
                        delta -= 1
                for k5_idx in triple_to_k5s[i]:
                    old_c = k5_blue_counts[k5_idx]
                    if old_c == 9:
                        delta += 1
            else: # Blue -> Red
                for k4_idx in triple_to_k4s[i]:
                    old_c = k4_red_counts[k4_idx]
                    if old_c == 3:
                        delta += 1
                for k5_idx in triple_to_k5s[i]:
                    old_c = k5_blue_counts[k5_idx]
                    if old_c == 10:
                        delta -= 1

            if delta < 0:
                colors[i] = new_color
                if new_color == 1:
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_counts[k4_idx] -= 1
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_counts[k5_idx] += 1
                else:
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_counts[k4_idx] += 1
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_counts[k5_idx] -= 1
                        
                total_violations += delta
                no_improve_steps = 0
                
                if total_violations == 0:
                    return {'n': target_n, 'colors': colors}
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    
            elif delta == 0:
                if random.random() < 0.1:
                    colors[i] = new_color
                    if new_color == 1:
                        for k4_idx in triple_to_k4s[i]:
                            k4_red_counts[k4_idx] -= 1
                        for k5_idx in triple_to_k5s[i]:
                            k5_blue_counts[k5_idx] += 1
                    else:
                        for k4_idx in triple_to_k4s[i]:
                            k4_red_counts[k4_idx] += 1
                        for k5_idx in triple_to_k5s[i]:
                            k5_blue_counts[k5_idx] -= 1
                    no_improve_steps += 1
            else:
                pass
                
            if total_violations < best_violations:
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            if no_improve_steps > 3000:
                break
                
    if best_violations == 0:
        return {'n': target_n, 'colors': best_colors}
        
    # Fallback
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
# EVOLVE-BLOCK-END
