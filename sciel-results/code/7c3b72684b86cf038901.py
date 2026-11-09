def construct():
    n = 256
    best_signs = None
    best_cost = float('inf')
    
    start_time = time.time()
    timeout = 55.0
    
    while (time.time() - start_time) < timeout:
        signs = [random.randint(0, 1) * 2 - 1 for _ in range(n)]
        current_cost = max_ap_discrepancy(signs)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_signs = signs[:]
            
        T = 2.0
        cooling_rate = 0.9995
        
        while (time.time() - start_time) < timeout:
            i = random.randint(0, n - 1)
            signs[i] *= -1
            
            new_cost = max_ap_discrepancy(signs)
            delta = new_cost - current_cost
            
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_signs = signs[:]
            else:
                if random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    signs[i] *= -1
            
            T *= cooling_rate
            if T < 0.1:
                break
                
    return best_signs
