def construct():
    es = set()
    # First K_{3,3}: vertices 0,1,2 connected to 3,4,5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            es.add(tuple(sorted((u, v))))
    # Second K_{3,3}: vertices 6,7,8 connected to 9,10,11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            es.add(tuple(sorted((u, v))))
    return [list(e) for e in sorted(es)]
