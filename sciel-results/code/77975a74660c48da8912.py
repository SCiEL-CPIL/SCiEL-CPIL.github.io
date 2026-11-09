import sys
import numpy as np

def construct():
    N = 5 * 10**7
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    solutions = {}
    best_M = 0
    best_count = 0
    best_k_list = []
    
    sol_get = solutions.get
    limit = 10**18
    
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= limit:
            lst = sol_get(M)
            if lst is None:
                solutions[M] = [k]
            else:
                lst.append(k)
                if len(lst) > best_count:
                    best_count = len(lst)
                    best_M = M
                    best_k_list = lst
                    
    return {'M': int(best_M), 'k': best_k_list}
