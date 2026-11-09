                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]: # S[:-1] is S before adding x
                    sums.add(x + a)
