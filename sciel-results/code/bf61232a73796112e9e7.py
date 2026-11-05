import math
import sys
import time

def construct():
    N = 256
    valid_values = []
    for a in range(1, N + 1):
        if not _is_square(2 * a):
            valid_values.append(a)
    
    M = len(valid_values)
    if M == 0:
        return [1]
        
    val_to_idx = {v: i for i, v in enumerate(valid_values)}
    
    squares = set()
    k = 1
    while k*k <= 2*N:
        squares.add(k*k)
        k += 1
        
    adj = [0] * M
    for i in range(M):
        vi = valid_values[i]
        for j in range(i + 1, M):
            vj = valid_values[j]
            if (vi + vj) in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Greedy Lower Bound
    P = (1 << M) - 1
    greedy_set = set()
    P_copy = P
    while P_copy:
        min_deg = M
        min_v = -1
        temp = P_copy
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (P_copy & adj[v]).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            temp ^= lsb
            
        if min_v == -1:
            break
            
        greedy_set.add(min_v)
        P_copy &= ~(adj[min_v] | (1 << min_v))
        
    best_count = len(greedy_set)
    best_set = set(greedy_set)
    
    # Branch and Bound
    sys.setrecursionlimit(10000)
    calls = 0
    limit = 2000000 # Safety limit
    
    def search(P, count, current_set):
        nonlocal best_count, best_set, calls
        
        calls += 1
        if calls > limit:
            return
            
        if count + P.bit_count() <= best_count:
            return
            
        if P == 0:
            if count > best_count:
                best_count = count
                best_set = set(current_set)
            return
            
        # Pick vertex with max degree in P
        max_deg = -1
        v = -1
        temp = P
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = (P & adj[idx]).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp ^= lsb
            
        if v == -1:
            return
            
        # Include v
        P_include = P & ~(adj[v] | (1 << v))
        current_set.add(v)
        search(P_include, count + 1, current_set)
        current_set.remove(v)
        
        # Exclude v
        P_exclude = P & ~(1 << v)
        search(P_exclude, count, current_set)
        
    initial_P = (1 << M) - 1
    search(initial_P, 0, set())
    
    result = [valid_values[i] for i in best_set]
    
    if not result:
        result = [valid_values[0]]
        
    if _verify(result):
        return sorted(result)
    else:
        return [1]

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _verify(s):
    s_set = set(s)
    if not s_set:
        return False
    for a in s_set:
        if _is_square(2 * a):
            return False
    for i, a in enumerate(s):
        for b in s[i+1:]:
            if _is_square(a + b):
                return False
    return True
