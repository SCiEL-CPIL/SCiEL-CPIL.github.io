        if k >= 3:
            for i in range(k):
                b = S[i]
                for j in range(i+1, k):
                    c = S[j]
                    for l in range(j+1, k):
                        d = S[l]
                        if (b + c + d) % x == 0:
                            return False
