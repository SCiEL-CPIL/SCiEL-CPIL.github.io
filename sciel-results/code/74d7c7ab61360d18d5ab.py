    def is_valid_insert(x):
        if sum_counts[2*x] >= 2:
            return False
        for y in S:
            if sum_counts[x + y] >= 2:
                return False
        return True
