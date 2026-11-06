import time
import math
import numpy as np

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit to be safe
    
    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    
    divisors.sort()
    
    # 2. Greedy Set Cover with NumPy
    cover_count = np.zeros(base, dtype=np.int8)
    used_moduli = set()
    classes = []
    
    while True:
        if time.time() > deadline - 5.0:
            break
            
        best_gain = -1
        best_m = None
        best_r = None
        
        zeros = (cover_count == 0)
        
        for m in divisors:
            if m in used_moduli:
                continue
            
            try:
                zeros_reshaped = zeros.reshape(-1, m)
                gains = zeros_reshaped.sum(axis=0)
                best_r_for_m = np.argmax(gains)
                best_gain_for_m = gains[best_r_for_m]
                
                if best_gain_for_m > best_gain:
                    best_gain = best_gain_for_m
                    best_m = m
                    best_r = best_r_for_m
            except:
                pass
            
        if best_m is None:
            break
            
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        cover_count[best_r::best_m] += 1
        
        if time.time() > deadline - 10.0:
            break

    # 3. Local Search Refinement
    local_search_deadline = start_time + 148.0
    improved = True
    
    while improved and time.time() < local_search_deadline:
        improved = False
        current_score = np.sum(cover_count == 0)
        
        for idx, (m_out, r_out) in enumerate(classes):
            if time.time() > local_search_deadline - 2.0:
                break
                
            cover_count[r_out::m_out] -= 1
            score_after_removal = np.sum(cover_count == 0)
            required_gain = score_after_removal - current_score + 1
            
            best_swap_gain = -1
            best_m_in = None
            best_r_in = None
            
            for m_in in divisors:
                if m_in in used_moduli:
                    continue
                
                try:
                    zeros_reshaped = (cover_count == 0).reshape(-1, m_in)
                    gains = zeros_reshaped.sum(axis=0)
                    best_r_for_m = np.argmax(gains)
                    best_gain_for_m = gains[best_r_for_m]
                    
                    if best_gain_for_m >= required_gain:
                        if best_gain_for_m > best_swap_gain:
                            best_swap_gain = best_gain_for_m
                            best_m_in = m_in
                            best_r_in = best_r_for_m
                except:
                    pass
            
            if best_m_in is not None:
                cover_count[best_r_in::best_m_in] += 1
                classes[idx] = [best_m_in, best_r_in]
                used_moduli.remove(m_out)
                used_moduli.add(best_m_in)
                improved = True
                break
            else:
                cover_count[r_out::m_out] += 1

    return classes
