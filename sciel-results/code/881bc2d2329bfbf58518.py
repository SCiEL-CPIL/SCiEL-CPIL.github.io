def _solve(D):
    req = set(range(1,31))
    S = [0]
    used = set()
    # used differences
    def dfs(S, used):
        if _time_left() < 0.2:
            return None
        remaining = req - used
        if not remaining:
            return list(S)
        if len(S) >= MAXN:
            return None
        # candidate x in 1..D not in S
        # compute gains
        cands = []
        for x in range(1, D+1):
            if x in S:
                continue
            nd = []
            ok = True
            for s in S:
                d = x - s
                if d < 0: d = -d
                if d in used:
                    ok = False
                    break
                nd.append(d)
            if not ok:
                continue
            gain = sum(1 for d in nd if d in remaining)
            if gain > 0:
                cands.append((gain, x, nd))
        if not cands:
            return None
        cands.sort(key=lambda t: (-t[0], t[1]))
        for gain, x, nd in cands:
            newS = S + [x]
            newUsed = used | set(nd)
            r = dfs(newS, newUsed)
            if r is not None:
                return r
        return None
    return dfs(S, used)
