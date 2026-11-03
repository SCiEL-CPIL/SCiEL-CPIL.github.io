    def block_rank(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return 7
        if a == 3 and b == 2 and c == 2:
            return 11
        if a == 2 and b == 3 and c == 2:
            return 11
        if a == 2 and b == 2 and c == 3:
            return 11
        # Other small improvements?
        # 3x3x2: 2x2x2 (7) + 1x2x2 (4) + 2x1x2 (4) + 1x1x2 (2) = 17? Schoolbook 18.
        # So 3x3x2 rank 17.
        if a == 3 and b == 3 and c == 2:
            return 17
        if a == 3 and b == 2 and c == 3:
            return 17
        if a == 2 and b == 3 and c == 3:
            return 17
        return a * b * c
