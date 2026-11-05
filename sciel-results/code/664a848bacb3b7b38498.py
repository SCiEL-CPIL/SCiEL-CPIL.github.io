    def get_cost(S_set, x):
        count = 0
        for (a, b) in conflicts_with[x]:
            if a in S_set and b in S_set:
                count += 1
        return count
