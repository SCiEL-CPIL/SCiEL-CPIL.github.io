import time

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110
    q = k
    d = g - 5 if g % 2 else g - 4
    d = max(3, d)
    
    if q == 4:
        _mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        _add = lambda x,y: x^y
        _sub = lambda x,y: x^y
        _mul = lambda x,y: _mul_table[x][y]
    else:
        _add = lambda x,y: (x+y)%q
        _sub = lambda x,y: (x-y)%q
        _mul = lambda x,y: x*y%q
    
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5)%4
        c0 = c-r
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    start = (0, (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i in range(len(order)):
        if time.time() > deadline:
            break
        side, x = order[i]
        for first in range(q):
            y = [0]*(d+1)
            y[1] = first
            if side == 0:
                for c in range(2, d+1):
                    a,b = rule[c]
                    y[c] = _add(x[c], _mul(y[a], x[b]))
            else:
                for c in range(2, d+1):
                    a,b = rule[c]
                    y[c] = _sub(x[c], _mul(x[a], y[b]))
            other = (1-side, tuple(y[1:]))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((i, index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
