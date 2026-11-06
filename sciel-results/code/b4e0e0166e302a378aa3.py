import time

def construct():
    deadline = time.time() + 85
    n = 128
    
    def verify(colors):
        """Verify all 4-term APs have >=3 distinct colors."""
        for a in range(1, n - 2):
            for d in range(1, (n - a) // 3 + 1):
                if a + 3 * d <= n:
                    cs = [colors[a-1], colors[a+d-1], colors[a+2*d-1], colors[a+3*d-1]]
                    if len(set(cs)) < 3:
                        return False
        return True
    
    # Try greedy with k colors
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        colors = []
        valid = True
        for x in range(1, n + 1):
            found = False
            for c in range(k):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a >= 1:
                        cs = [colors[a-1], colors[a+d-1], colors[a+2*d-1], c]
                        if len(set(cs)) < 3:
                            ok = False
                            break
                if ok:
                    colors.append(c)
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid and verify(colors):
            return colors
    
    # Fallback: seed-style greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
