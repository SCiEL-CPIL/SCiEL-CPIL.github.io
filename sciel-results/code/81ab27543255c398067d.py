    for iteration in range(30):
        ...
            if ok:
                old_mask = 0
                ...
                if not (new_mask & remaining):
                    used_bits = (used_bits & ~old_mask) | new_mask
