import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_M = 0
    best_k_list = []
    
    def search(limit):
        sigma = np.zeros(limit + 1, dtype=np.int64)
        for i in range(1, limit + 1):
            sigma[i::i] += i
            
        ks = np.arange(1, limit + 1, dtype=np.int64)
        Ms = ks * sigma[1:]
        
        # Filter if necessary (M <= 10^18)
        # For limit <= 10^8, M is small enough.
        
        order = np.argsort(Ms)
        sorted_Ms = Ms[order]
        sorted_ks = ks[order]
        
        best_M_local = 0
        best_k_local = []
        max_count = 0
        
        i = 0
        n = len(sorted_Ms)
        while i < n:
            current_M = sorted_Ms[i]
            count = 0
            k_list = []
            while i < n and sorted_Ms[i] == current_M:
                count += 1
                k_list.append(int(sorted_ks[i]))
                i += 1
            
            if count > max_count:
                max_count = count
                best_M_local = int(current_M)
                best_k_local = k_list
                
        return best_M_local, best_k_local

    current_limit = 10**5
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        M, k_list = search(current_limit)
        
        if len(k_list) > len(best_k_list):
            best_M = M
            best_k_list = k_list
            
        # Increase limit
        if current_limit < 10**6:
            current_limit = int(current_limit * 2)
        elif current_limit < 5 * 10**6:
            current_limit = int(current_limit * 1.5)
        elif current_limit < 10**7:
            current_limit = int(current_limit * 1.2)
        else:
            break
            
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_k_list}
