        for i in range(K):
            xi = S_list[i]
            for j in range(i + 1, K):
                xj = S_list[j]
                target = (-xi - xj) % v
                if target in S_set_copy:
                    # Ensure distinctness: target != xi and target != xj
                    if target != xi and target != xj:
                        return False
