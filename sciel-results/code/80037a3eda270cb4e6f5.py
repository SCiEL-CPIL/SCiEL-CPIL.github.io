import random
import math
import time

def construct():
    target_min = 128
    target_max = 256
    n = 128
    
    def calculate_cost_fast(A_list):
        k = len(A_list)
        if k == 0:
            return False, float('inf')
        
        counts = [0] * 257
        
        for i in range(k):
            a = A_list[i]
            for j in range(i, k):
                b = A_list[j]
                s = a + b
                if target_min <= s <= target_max:
                    counts[s] += 1
        
        max_count = 0
        for s in range(target_min, target_max + 1):
            c = counts[s]
            if c == 0:
                return False, float('inf')
            if c > max_count:
                max_count = c
                
        return True, max_count

    best_A_global = None
    best_cost_global = float('inf')
    
    start_time_total = time.time()
    time_limit_total = 55.0
    
    for k in [16, 17, 18, 19, 20]:
        if time.time() - start_time_total > time_limit_total - 5:
            break
            
        # Initialize
        A = random.sample(range(n + 1), k)
        # Basic feasibility check
        if min(A) > 64 or max(A) < 128:
            # Try to fix
            if min(A) > 64:
                A[random.randint(0, k-1)] = random.randint(0, 64)
            if max(A) < 128:
                A[random.randint(0, k-1)] = 128
        
        valid, cost = calculate_cost_fast(A)
        current_cost = cost if valid else 1000 + cost
        best_A_local = A.copy()
        best_cost_local = current_cost
        
        T = 5.0
        cooling = 0.9995
        start_time_sa = time.time()
        time_limit_sa = 4.0
        
        while time.time() - start_time_sa < time_limit_sa:
            idx = random.randint(0, k - 1)
            old_val = A[idx]
            new_val = random.randint(0, n)
            
            if new_val == old_val:
                continue
                
            if new_val in A:
                continue
                
            A[idx] = new_val
            
            new_valid, new_cost = calculate_cost_fast(A)
            new_actual_cost = new_cost if new_valid else 1000 + new_cost
            
            if new_actual_cost < current_cost:
                accept = True
            else:
                delta = new_actual_cost - current_cost
                if T > 0:
                    accept = random.random() < math.exp(-delta / T)
                else:
                    accept = False
            
            if accept:
                current_cost = new_actual_cost
                if new_actual_cost < best_cost_local:
                    best_cost_local = new_actual_cost
                    best_A_local = A.copy()
                    if best_cost_local == 2:
                        break
            else:
                A[idx] = old_val
                
            T *= cooling
        
        if best_cost_local < best_cost_global:
            best_cost_global = best_cost_local
            best_A_global = best_A_local.copy()
            
        if best_cost_global == 2:
            break

    if best_A_global is None:
        return list(range(64, 129))
        
    return best_A_global
