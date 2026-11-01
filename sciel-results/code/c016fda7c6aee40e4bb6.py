    def generate_walks():
        n = 6
        edge_list = [(u,v) for u in range(n) for v in range(u+1, n)]
        edge_idx = {e:i for i,e in enumerate(edge_list)}
        unique_walks = set()
        
        def dfs(start, cur, prv, steps_left, coeffs):
            if steps_left == 0:
                if cur == start:
                    unique_walks.add(tuple(coeffs))
                return
            
            for nv in range(n):
                if nv == cur or nv == prv:
                    continue
                u, v = min(cur, nv), max(cur, nv)
                idx = edge_idx[(u,v)]
                sign = 1 if cur < nv else -1
                coeffs[idx] += sign
                dfs(start, nv, cur, steps_left - 1, coeffs)
                coeffs[idx] -= sign
        
        for L in range(3, 9):
            for start in range(n):
                for v1 in range(n):
                    if v1 == start: continue
                    u, v = min(start, v1), max(start, v1)
                    idx = edge_idx[(u,v)]
                    sign = 1 if start < v1 else -1
                    coeffs = [0]*15
                    coeffs[idx] += sign
                    dfs(start, v1, start, L-1, coeffs)
        
        return np.array(list(unique_walks), dtype=np.int32)
    