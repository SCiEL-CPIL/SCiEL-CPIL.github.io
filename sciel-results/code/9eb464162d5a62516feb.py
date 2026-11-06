import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 128
    
    pos_aps = [[] for _ in range(n)]
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            quad = (a, a + d, a + 2 * d, a + 3 * d)
            for i in range(4):
                others = tuple(quad[j] for j in range(4) if j != i)
                pos_aps[quad[i]].append(others)
    
    colors = [-1] * n
    colors[0] = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        for c in range(n):
            valid = True
            for others in pos_aps[i]:
                c0, c1, c2 = colors[others[0]], colors[others[1]], colors[others[2]]
                if c0 != -1 and c1 != -1 and c2 != -1:
                    if c0 == c1 == c2:
                        if c == c0:
                            valid = False
                            break
                    else:
                        s = {c0, c1, c2}
                        if len(s) == 2:
                            if c in s:
                                valid = False
                                break
                        elif len(s) == 1:
                            if c == c0:
                                valid = False
                                break
                        # len(s) == 3: any c works
            if valid:
                colors[i] = c
                break
        else:
            colors[i] = 0
    
    return colors
