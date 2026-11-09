def construct():
    edges = set()
    # First K_{3,3}: parts {0,1,2} and {3,4,5}
    for a in (0,1,2):
        for b in (3,4,5):
            edges.add(tuple(sorted((a,b))))
    # Second K_{3,3}: parts {6,7,8} and {9,10,11}
    for a in (6,7,8):
        for b in (9,10,11):
            edges.add(tuple(sorted((a,b))))
    return [list(e) for e in sorted(edges)]
