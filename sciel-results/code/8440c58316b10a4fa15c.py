        def calc_mask(row):
            m = 0
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    m |= 1 << (row[j] - row[i])
            return m
        