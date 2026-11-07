import time

def _valid(s):
    S = set(s)
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            ab = a+b
            if ab not in S:
                continue
            for k in range(j+1, n):
                c = lst[k]
                if (a+c) in S and (b+c) in S:
                    return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    base = set(range(17,33)) | set(range(65,129))
    cur = sorted(base)
    best = cur
    # greedy add
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1,129):
            if x in cur: continue
            cand = cur + [x]
            if _valid(cand):
                cur = cand
                best = cur
                changed = True
                break
    return [int(v) for v in best]
