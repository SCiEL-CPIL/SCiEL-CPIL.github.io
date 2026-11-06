def priority(vector, n):
        count_2 = 0
        for x in vector:
            if x == 2:
                count_2 += 1
        return -float(count_2)
