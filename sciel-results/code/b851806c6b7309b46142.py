import math
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85
    
    max_n = 10210
    kernels = [1] * (max_n + 1)
    
    def squarefree_kernel(n):
        res = 1
        i = 2
        while i * i <= n:
            if n % i == 0:
                cnt = 0
                while n % i == 0:
                    n //= i
                    cnt += 1
                if cnt % 2 == 1:
                    res *= i
            i += 1
        if n > 1:
            res *= n
        return res

    for i in range(2, max_n + 1):
        kernels[i] = squarefree_kernel(i)
        
    target = 210
    
    def can_form_solution(M):
        candidates = list(range(211, M + 1))
        if not candidates:
            return None
        
        cand_kernels = [kernels[x] for x in candidates]
        n_cands = len(candidates)
        
        # Precompute suffix OR of kernels
        suffix_or = [0] * (n_cands + 1)
        for i in range(n_cands - 1, -1, -1):
            suffix_or[i] = suffix_or[i + 1] | cand_kernels[i]
            
        selected = []
        
        def dfs(idx, current_xor):
            if time.time() > deadline:
                return None
            
            if current_xor == target:
                return list(selected)
                
            if idx >= n_cands:
                return None
                
            req = current_xor ^ target
            # If req has bits not in the remaining candidates, impossible
            if req & ~suffix_or[idx]:
                return None
            
            # Option 1: Skip
            res = dfs(idx + 1, current_xor)
            if res is not None:
                return res
                
            # Option 2: Include
            selected.append(candidates[idx])
            res = dfs(idx + 1, current_xor ^ cand_kernels[idx])
            if res is not None:
                return res
            selected.pop()
            
            return None
            
        return dfs(0, 1)

    for M in range(211, 10211):
        if time.time() > deadline:
            break
        sol = can_form_solution(M)
        if sol is not None:
            # Verify
            prod = 210
            for x in sol:
                prod *= x
            if math.isqrt(prod) ** 2 == prod:
                return [int(x) for x in sol]
            else:
                return [int(x) for x in sol]
                
    return [224, 240]

# EVOLVE-BLOCK-END
