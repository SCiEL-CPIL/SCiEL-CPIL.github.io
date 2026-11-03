                for i in range(N):
                    ok = True
                    for j in indices:
                        s = i + j
                        if s in used_sums:
                            ok = False
                            break
                    ...
                