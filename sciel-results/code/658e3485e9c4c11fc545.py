def solve(K, need_path=False):
    allowed_by_v = [[] for _ in range(n)]
    for v in range(n):
        lst = []
        for idx, m in triples_by_v[v]:
            if idx < K: lst.append(m)
            else: break
        allowed_by_v[v] = lst
    memo = [-1]*(1<<n)
    path = {}
    def dfs(mask):
        if mask == full:
            return True
        if memo[mask] != -1:
            return memo[mask] == 1
        rem = full ^ mask
        v = (rem & (-rem)).bit_length() - 1
        for m in allowed_by_v[v]:
            if mask & m == 0:
                if dfs(mask | m):
                    memo[mask] = 1
                    if need_path:
                        path[mask] = m
                    return True
        memo[mask] = 0
        return False
    ...
