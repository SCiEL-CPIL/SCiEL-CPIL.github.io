def batcher_oom(n):
    comps = []
    def oem(a, b, n):
        if n == 1:
            comps.append((a, b))
        else:
            oem(a, b, n//2)
            oem(a+1, b+1, n//2)
            for i in range(n-1):
                comps.append((a+1+i, b+1+i))
    def sort(a, n):
        if n == 1: return
        if n == 2:
            comps.append((a, a+1)); return
        sort(a, n//2)
        sort(a+n//2, n//2)
        oem(a, a+n//2, n//2)
    sort(0, n)
    return comps
