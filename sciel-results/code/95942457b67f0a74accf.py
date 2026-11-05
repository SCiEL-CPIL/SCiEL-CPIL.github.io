def _precompute_dp(n_max, m_max, p_max):
    dp_rank = {}
    dp_split = {}
    
    # Initialize with schoolbook
    for n in range(1, n_max + 1):
        for m in range(1, m_max + 1):
            for p in range(1, p_max + 1):
                dp_rank[(n, m, p)] = n * m * p
                dp_split[(n, m, p)] = None # None means schoolbook
                
    # Iterate in increasing order of size?
    # Since splits are smaller, we can just iterate n, m, p from 1 to max
    # But dp(n,m,p) depends on dp(n1,m1,p1) which is smaller.
    # So simple iteration 1..max works.
    
    for n in range(1, n_max + 1):
        for m in range(1, m_max + 1):
            for p in range(1, p_max + 1):
                current_best = dp_rank[(n, m, p)]
                
                for n1 in range(1, n):
                    n2 = n - n1
                    for m1 in range(1, m):
                        m2 = m - m1
                        for p1 in range(1, p):
                            p2 = p - p1
                            
                            rank = dp_rank[(n1, m1, p1)] + dp_rank[(n1, m1, p2)] + dp_rank[(n1, m2, p1)] + dp_rank[(n1, m2, p2)] + \
                                   dp_rank[(n2, m1, p1)] + dp_rank[(n2, m1, p2)] + dp_rank[(n2, m2, p1)] + dp_rank[(n2, m2, p2)]
                            
                            if rank < current_best:
                                current_best = rank
                                dp_split[(n, m, p)] = (n1, m1, p1)
                
                dp_rank[(n, m, p)] = current_best
                
    return dp_rank, dp_split
