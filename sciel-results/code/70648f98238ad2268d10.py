    new_tail = []
    new_tail_set = set()
    for dx, dy in new_rel_tail:
        p = (pivot[0] + dx, pivot[1] + dy)
        if p in head_set or p in new_tail_set:
            return None
        new_tail_set.add(p)
        new_tail.append(p)
