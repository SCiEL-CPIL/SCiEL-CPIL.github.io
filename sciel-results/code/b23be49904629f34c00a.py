    def construct():
        n, m, p = DIMENSIONS
        start = time.time()
        deadline = start + 100.0
        
        # Try rank 45
        res = _try_rank_45(n, m, p, deadline)
        if res:
            U, V, W, rank = res
        else:
            U, V, W, rank = _schoolbook(n, m, p)
            
        def to_list(M):
            result = []
            for row in M:
                new_row = []
                for x in row:
                    if x.denominator == 1:
                        new_row.append(int(x))
                    else:
                        new_row.append(float(x))
                result.append(new_row)
            return result
            
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    