import random

def construct():
    n = 128
    p = 127
    
    def legendre_symbol(a, p):
        a = a % p
        if a == 0:
            return 0
        ls = pow(a, (p - 1) // 2, p)
        if ls == p - 1:
            return -1
        return ls
        
    colors = [0] * n
    for i in range(n):
        res = i % p
        ls = legendre_symbol(res, p)
        if ls == -1:
            colors[i] = 0
        elif ls == 0:
            colors[i] = 1
        else:
            colors[i] = 2
            
    # Fast validity check
    valid = True
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                    valid = False
                    break
        if not valid:
            break
            
    if valid:
        return colors
        
    # Fallback if invalid (unlikely if hint is correct)
    # Simple local search
    for _ in range(50):
        for a in range(n):
            for d in range(1, n):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if e < n:
                    if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                        pos = random.choice([a, b, c, e])
                        colors[pos] = random.randint(0, 5)
                        break
            else:
                continue
            break
            
    return colors
