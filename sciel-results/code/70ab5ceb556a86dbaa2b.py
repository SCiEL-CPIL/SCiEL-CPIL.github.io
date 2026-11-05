def get_4set_masks(n):
    masks = []
    for combo in itertools.combinations(range(n), 4):
        m = 0
        for v in combo:
            m |= (1 << v)
        masks.append(m)
    return masks
