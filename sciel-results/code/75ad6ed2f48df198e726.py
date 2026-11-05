   def get_min_rank(si, sj, sk):
       if si == 1 and sj == 1 and sk == 1: return 1
       if si == 2 and sj == 2 and sk == 2: return 7
       best = si * sj * sk
       # Try splitting si
       if si > 1:
           for k in range(1, si):
               r1 = get_min_rank(k, sj, sk)
               r2 = get_min_rank(si-k, sj, sk)
               best = min(best, r1 + r2)
       # Try splitting sj
       if sj > 1:
           for k in range(1, sj):
               r1 = get_min_rank(si, k, sk)
               r2 = get_min_rank(si, sj-k, sk)
               best = min(best, r1 + r2)
       # Try splitting sk
       if sk > 1:
           for k in range(1, sk):
               r1 = get_min_rank(si, sj, k)
               r2 = get_min_rank(si, sj, sk-k)
               best = min(best, r1 + r2)
       return best
   