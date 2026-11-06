import random
import time

def construct():
    N = 256
    c = [random.randint(0, 1) for _ in range(N)]
    
    def get_color(idx_1based):
        return c[idx_1based - 1]
        
    def set_color(idx_1based, val):
        c[idx_1based - 1] = val

    def count_mono_4aps():
        count = 0
        max_d = (N - 1) // 3
        for d in range(1, max_d + 1):
            max_a = N - 3 * d
            for a in range(1, max_a + 1):
                i1 = a
                i2 = a + d
                i3 = a + 2 * d
                i4 = a + 3 * d
                c1 = get_color(i1)
                c2 = get_color(i2)
                c3 = get_color(i3)
                c4 = get_color(i4)
                if c1 == c2 == c3 == c4:
                    count += 1
        return count

    current_count = count_mono_4aps()
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() - start_time < 110:
        i = random.randint(1, N)
        old_c = get_color(i)
        new_c = 1 - old_c
        
        delta = 0
        
        for k in range(4):
            max_d1 = (i - 1) // k if k > 0 else N
            max_d2 = (N - i) // (3 - k)
            max_d = min(max_d1, max_d2)
            
            for d in range(1, max_d + 1):
                a = i - k * d
                idxs = [a, a + d, a + 2 * d, a + 3 * d]
                
                other_idxs = [idx for idx in idxs if idx != i]
                
                cur_other_all_old = True
                new_other_all_new = True
                
                for idx in other_idxs:
                    col = get_color(idx)
                    if col != old_c:
                        cur_other_all_old = False
                    if col != new_c:
                        new_other_all_new = False
                    if not cur_other_all_old and not new_other_all_new:
                        break
                
                if cur_other_all_old:
                    delta -= 1
                if new_other_all_new:
                    delta += 1
        
        if delta < 0:
            set_color(i, new_c)
            current_count += delta
            no_improve_count = 0
        else:
            # Accept with small probability if delta == 0 to explore
            if delta == 0 and random.random() < 0.1:
                set_color(i, new_c)
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            c = [random.randint(0, 1) for _ in range(N)]
            current_count = count_mono_4aps()
            no_improve_count = 0

    return c
