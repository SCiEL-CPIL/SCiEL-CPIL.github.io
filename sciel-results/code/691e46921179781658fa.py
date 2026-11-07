    for sq in [sq1, sq2]:
        for r in range(n):
            if sorted(sq[r]) != list(range(n)):
                return [sq1] # Fallback to single square if broken
        for c in range(n):
            col = [sq[r][c] for r in range(n)]
            if sorted(col) != list(range(n)):
                return [sq1]
