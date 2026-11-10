        def _is_valid_addition_fast(S_set, x):
            S_list = list(S_set)
            n = len(S_list)
            
            # Check 1: x does not divide sum of any 3 in S_set
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        s = S_list[i] + S_list[j] + S_list[k]
                        if s % x == 0:
                            return False
            
            # Check 2: for each a in S_set, a does not divide sum of x + 2 others from S_set
            for i, a in enumerate(S_list):
                for j in range(n):
                    if j == i:
                        continue
                    for k in range(j + 1, n):
                        if k == i:
                            continue
                        s = x + S_list[j] + S_list[k]
                        if s % a == 0:
                            return False
            
            return True
        