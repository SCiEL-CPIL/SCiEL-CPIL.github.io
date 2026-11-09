import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def has_sink_flat(A, n, i, j, k, l):
        # A is flat list, index u*n + v
        # Check i
        if A[i*n+j] and A[i*n+k] and A[i*n+l]:
            return True
        # Check j
        if A[j*n+i] and A[j*n+k] and A[j*n+l]:
            return True
        # Check k
        if A[k*n+i] and A[k*n+j] and A[k*n+l]:
            return True
        # Check l
        if A[l*n+i] and A[l*n+j] and A[l*n+k]:
            return True
        return False

    def try_find_valid(n, max_time):
        # Init
        A = [0] * (n*n)
        for i in range(n):
            for j in range(i+1, n):
                val = random.randint(0, 1)
                A[i*n+j] = val
                A[j*n+i] = 1 - val
        
        bad_count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if not has_sink_flat(A, n, i, j, k, l):
                            bad_count += 1
        
        if bad_count == 0:
            return A
        
        no_improve_count = 0
        while time.time() < max_time:
            improved = False
            for _ in range(200): # More attempts per loop
                if time.time() >= max_time:
                    break
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                
                delta = 0
                # Iterate k,l
                # To speed up, we can skip if k,l are such that the set is known to be good/bad?
                # No, we just calculate.
                
                for k in range(n):
                    if k == i or k == j:
                        continue
                    for l in range(n):
                        if l == i or l == j or l == k:
                            continue
                        
                        # We can break early if delta becomes large positive?
                        # No, we need exact delta for 0 case.
                        
                        sink_before = has_sink_flat(A, n, i, j, k, l)
                        
                        # Flip
                        idx_ij = i*n+j
                        idx_ji = j*n+i
                        old_ij = A[idx_ij]
                        old_ji = A[idx_ji]
                        A[idx_ij] = 1 - old_ij
                        A[idx_ji] = 1 - old_ji
                        
                        sink_after = has_sink_flat(A, n, i, j, k, l)
                        
                        # Undo
                        A[idx_ij] = old_ij
                        A[idx_ji] = old_ji
                        
                        delta += (sink_after - sink_before)
                
                if delta > 0:
                    A[i*n+j] = 1 - A[i*n+j]
                    A[j*n+i] = 1 - A[j*n+i]
                    bad_count -= delta
                    improved = True
                elif delta == 0:
                    if random.random() < 0.1:
                        A[i*n+j] = 1 - A[i*n+j]
                        A[j*n+i] = 1 - A[j*n+i]
                        improved = True
                
            if bad_count == 0:
                return A
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 5:
                    break
            else:
                no_improve_count = 0
        
        return None

    A = try_find_valid(67, deadline)
    if A is not None:
        orientation = []
        for i in range(67):
            for j in range(i+1, 67):
                orientation.append(1 if A[i*n+j] else 0) # Wait, n is 67
        return {'n': 67, 'orientation': orientation}
    
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
