def construct():
    from itertools import combinations
    covered = set()
    blocks = []
    for b in combinations(range(12), 6):
        if not any(tuple(s) in covered for s in combinations(b, 5)):
            blocks.append(list(b))
            covered.update(combinations(b, 5))
    return blocks
