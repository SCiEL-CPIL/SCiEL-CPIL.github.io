import random
import time

def construct():
    deadline = time.time() + 115
    rng = random.Random(12345)
    
    def has_k4(n, A):
        for u in range(n):
            for v in range(u+1, n):
                if not (A[v] >> u) & 1: continue
                common = A[u] & A[v] & ~(1<<u) & ~(1<<v)
                c = common
                while c:
                    low = c & -c
                    w = low.bit_length() - 1
                    c ^= low
                    if (A[w] & (common ^ low)):
                        return True
        return False

    def find_k4(n, A):
        for u in range(n):
            for v in range(u+1, n):
                if not (A[v] >> u) & 1: continue
                common = A[u] & A[v] & ~(1<<u) & ~(1<<v)
                c = common
                while c:
                    low = c & -c
                    w = low.bit_length() - 1
                    c ^= low
                    rest = common ^ low
                    if (A[w] & rest):
                        low2 = (A[w] & rest) & -(A[w] & rest)
                        x = low2.bit_length() - 1
                        return (u,v,w,x)
        return None

    def find_ind6(n, A):
        comp = [0]*n
        for i in range(n):
            comp[i] = ((1<<n)-1) & ~(A[i] | (1<<i))
        
        found = [None]
        
        def bk(R, P, X):
            if found[0] is not None: return
            if len(R) == 6:
                found[0] = list(R)
                return
            if not P: return
            if len(R) + bin(P).count('1') < 6: return
            
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                candidates = P
                
            c = candidates
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
                
        bk([], (1<<n)-1, 0)
        return found[0]

    def valid(n, A):
        return not has_k4(n, A) and find_ind6(n, A) is None

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    # Start with Paley(17)
    q = 17
    squares = {x*x % q for x in range(1, q)}
    n = q
    A = [0]*n
    for i in range(n):
        for j in range(n):
            if i != j and (j-i) % q in squares:
                A[i] |= (1<<j)
                
    if not valid(n, A):
        # Fallback
        return {'n': 1, 'edges': []}
        
    best_n = n
    best_A = list(A)
    
    while time.time() < deadline - 5:
        new_n = best_n + 1
        if new_n > 40: break
        
        success = False
        for attempt in range(200):
            if time.time() > deadline - 5: break
            
            # Greedy triangle-free set N
            N = 0
            verts = list(range(best_n))
            rng.shuffle(verts)
            for v in verts:
                C = N & A[v]
                c = C
                has_edge = False
                while c:
                    low = c & -c
                    u = low.bit_length() - 1
                    c ^= low
                    if (A[u] & (C ^ low)):
                        has_edge = True
                        break
                if not has_edge:
                    N |= (1 << v)
            
            # Local search on N
            for step in range(100):
                if time.time() > deadline - 5: break
                
                A_new = best_A + [0]
                for i in range(best_n):
                    if (N >> i) & 1:
                        A_new[i] |= (1 << new_n - 1)
                        A_new[new_n - 1] |= (1 << i)
                
                if has_k4(new_n, A_new):
                    continue
                
                ind6 = find_ind6(new_n, A_new)
                if ind6 is None:
                    best_n = new_n
                    best_A = A_new
                    success = True
                    break
                
                # Repair: move a vertex from ind6 (excluding new vertex) to N
                candidates = [v for v in ind6 if v != new_n - 1]
                rng.shuffle(candidates)
                
                moved = False
                for v in candidates:
                    N_new = N | (1 << v)
                    # Check triangle-free
                    C = N_new & A[v]
                    c = C
                    has_edge = False
                    while c:
                        low = c & -c
                        u = low.bit_length() - 1
                        c ^= low
                        if (A[u] & (C ^ low)):
                            has_edge = True
                            break
                    if has_edge:
                        continue
                    
                    # Check validity with N_new
                    A_temp = best_A + [0]
                    for i in range(best_n):
                        if (N_new >> i) & 1:
                            A_temp[i] |= (1 << new_n - 1)
                            A_temp[new_n - 1] |= (1 << i)
                            
                    if has_k4(new_n, A_temp):
                        continue
                        
                    if find_ind6(new_n, A_temp) is None:
                        best_n = new_n
                        best_A = A_temp
                        success = True
                        moved = True
                        break
                    else:
                        # Accept N_new as next state for local search
                        N = N_new
                        break
                if moved or success:
                    break
                if success:
                    break
                    
            if success:
                break
        
        if not success:
            break

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
