import time

def construct():
    N = 128
    deadline = time.time() + 50
    
    for nc in range(3, 12):
        if time.time() > deadline - 5:
            break
        res = _bt(N, nc, deadline)
        if res is not None:
            return res
    
    return _greedy(N, deadline)

def _bt(N, nc, deadline):
    colors = [-1] * (N + 1)
    
    def check(pos, c):
        for d in range(1, (pos - 1) // 3 + 1):
            a = pos - 3 * d
            if a < 1:
                break
            c1, c2, c3 = colors[a], colors[a+d], colors[a+2*d]
            if c1 != -1 and c2 != -1 and c3 != -1:
                if len({c1, c2, c3, c}) < 3:
                    return False
        return True
    
    def bt(pos):
        if time.time() > deadline:
            return False
        if pos > N:
            return True
        for c in range(nc):
            if check(pos, c):
                colors[pos] = c
                if bt(pos + 1):
                    return True
                colors[pos] = -1
        return False
    
    if bt(1):
        return [colors[i] for i in range(1, N + 1)]
    return None

def _greedy(N, deadline):
    colors = [-1] * (N + 1)
    for pos in range(1, N + 1):
        if time.time() > deadline:
            colors[pos] = 0
            continue
        for c in range(128):
            ok = True
            for d in range(1, (pos - 1) // 3 + 1):
                a = pos - 3 * d
                if a < 1:
                    break
                c1, c2, c3 = colors[a], colors[a+d], colors[a+2*d]
                if c1 != -1 and c2 != -1 and c3 != -1:
                    if len({c1, c2, c3, c}) < 3:
                        ok = False
                        break
            if ok:
                colors[pos] = c
                break
        else:
            colors[pos] = 0
    return [colors[i] for i in range(1, N + 1)]
