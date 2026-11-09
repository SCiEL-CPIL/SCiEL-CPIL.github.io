def construct():
    M = 10
    # k from 2 to 11
    ks = list(range(2, 12))
    classes = []
    for i, k in enumerate(ks):
        m = M * k
        r = i  # residues 0, 1, ..., 9
        classes.append([m, r])
    return classes
