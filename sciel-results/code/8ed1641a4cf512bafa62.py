import time
import random

def construct():
    start = time.time()
    deadline = start + 115
    
    def get_in_mask(orient, n):
        in_mask = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orient[idx] == 0:
                    in_mask[j] |= (1 << i)
                else:
                    in_mask[i] |= (1 << j)
                idx += 1
        return in_mask
    
    def find_violation(in_mask, n):
        for a in range(n):
            for b in range(a+1, n):
                ab = in_mask[a] & in_mask[b]
                for c in range(b+1, n):
                    abc = ab & in_mask[c]
                    for d in range(c+1, n):
                        abcd = abc & in_mask[d]
                        s_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        if not (abcd & ~s_mask):
                            return (a, b, c, d)
        return None
    
    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i*(n-1) - i*(i-1)//2 + (j-i-1)
    
    def flip_edge(orient, in_mask, n, i, j):
        if i > j:
            i, j = j, i
        idx = edge_index(i, j, n)
        orient[idx] = 1 - orient[idx]
        if orient[idx] == 0:
            in_mask[j] |= (1 << i)
            in_mask[i] &= ~(1 << j)
        else:
            in_mask[i] |= (1 << j)
            in_mask[j] &= ~(1 << i)
    
    def paley_tournament(p):
        squares = set(x*x % p for x in range(1, p))
        orient = []
        for i in range(p):
            for j in range(i+1, p):
                if (j - i) % p in squares:
                    orient.append(0)
                else:
                    orient.append(1)
        return orient
    
    # Check Paley for small primes
    for p in [31, 43, 59, 67, 71, 79]:
        if time.time() > deadline - 20:
            break
        if p % 4 == 3:
            orient = paley_tournament(p)
            in_mask = get_in_mask(orient, p)
            if find_violation(in_mask, p) is None:
                return {'n': p, 'orientation': orient}
    
    # Random search with local improvement
    for n in range(31, 68):
        if time.time() > deadline - 15:
            break
        
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            
            orient = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
            in_mask = get_in_mask(orient, n)
            
            for step in range(500):
                if time.time() > deadline - 10:
                    break
                viol = find_violation(in_mask, n)
                if viol is None:
                    return {'n': n, 'orientation': orient}
                
                a, b, c, d = viol
                # Try flipping edges to fix
                fixed = False
                for v in (a, b, c, d):
                    for u in range(n):
                        if u == v:
                            continue
                        # Try flipping edge (v, u)
                        flip_edge(orient, in_mask, n, v, u)
                        if find_violation(in_mask, n) is None:
                            fixed = True
                            break
                        flip_edge(orient, in_mask, n, v, u)  # undo
                    if fixed:
                        break
                if not fixed:
                    # Random restart
                    orient = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
                    in_mask = get_in_mask(orient, n)
    
    # Fallback: Paley 83
    p = 83
    orient = paley_tournament(p)
    return {'n': p, 'orientation': orient}
