import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def solve_window(L, R):
        if time.time() > deadline:
            return None
        
        options = []
        for d in divisors:
            first = ((L + d - 1) // d) * d
            opts = []
            v = first
            while v <= R:
                opts.append(v)
                v += d
            options.append(opts)
        
        indices = sorted(range(n), key=lambda i: len(options[i]))
        
        if any(len(options[i]) == 0 for i in range(n)):
            return None
        
        used = set()
        result = [None] * n
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            i = indices[pos]
            for val in options[i]:
                if val not in used:
                    used.add(val)
                    result[i] = val
                    if backtrack(pos + 1):
                        return True
                    used.remove(val)
                    result[i] = None
            
            return False
        
        if backtrack(0):
            return result
        return None
    
    best = None
    best_width = float('inf')
    
    for L in range(1, 3001):
        if time.time() > deadline:
            break
        
        for W in range(10, best_width):
            if time.time() > deadline:
                break
            
            R = L + W
            if R > 1000000:
                break
            
            sol = solve_window(L, R)
            if sol is not None:
                if len(set(sol)) == n:
                    valid = all(sol[j] % divisors[j] == 0 for j in range(n))
                    if valid:
                        best = sol[:]
                        best_width = W
                        break
    
    if best is None:
        best = divisors[:]
    
    return [int(x) for x in best]
