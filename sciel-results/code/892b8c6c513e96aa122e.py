        x = rng.randint(lo, hi)
        new = {x - a for a in row}
        if not (new & used) and not (new & local):
            row.append(x)
            local |= new
    if len(row) == k + 1:
        return row, local
    return None
