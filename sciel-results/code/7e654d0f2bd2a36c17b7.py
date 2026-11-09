import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    colors = []
    
    for x in range(1, n + 1):
        for c in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                p0 = x - 3 * d
                if p0 < 1:
                    continue
                c0 = colors[p0 - 1]
                c1 = colors[x - 2 * d - 1]
                c2 = colors[x - d - 1]
                distinct = len({c0, c1, c2, c})
                if distinct < 3:
                    valid = False
                    break
            if valid:
                colors.append(c)
                break
        else:
            colors.append(0)
    
    return colors
