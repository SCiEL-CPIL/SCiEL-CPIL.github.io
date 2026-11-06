def _check_ap_validity(colors, ap, k):
    a, b, c, e = ap
    ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
    
    assigned = [x for x in (ca, cb, cc, ce) if x != -1]
    unassigned_count = 4 - len(assigned)
    
    if unassigned_count == 0:
        return len(set(assigned)) >= 3
    
    if unassigned_count == 1:
        s = set(assigned)
        # We add 1 color. Max size is len(s) + 1.
        # We need len(s) + 1 >= 3 => len(s) >= 2.
        if len(s) < 2:
            return False
        # If len(s) == 2, we need to pick a color not in s. Possible if K > 2.
        # If len(s) == 3, we can pick any color.
        if len(s) == 2:
            return K > 2
        return True

    if unassigned_count == 2:
        s = set(assigned)
        # We add 2 colors. Max size is len(s) + 2.
        # We need len(s) + 2 >= 3 => len(s) >= 1.
        # If len(s) == 1, we need to add 2 distinct colors not in s.
        # Possible if K > 1 + 2 = 3? No, we need 2 distinct colors from K-1 available.
        # So we need K-1 >= 2 => K >= 3.
        if len(s) == 1:
            return K >= 3
        # If len(s) >= 2, we can always satisfy.
        # E.g. s={0,1}, add 0,0 -> {0,1} size 2. Bad.
        # We need final size >= 3.
        # If s={0,1}, we need at least one new color.
        # We have 2 slots. We can put one new color.
        # So possible if K > 2.
        if len(s) == 2:
            return K > 2
        return True

    if unassigned_count == 3:
        s = set(assigned)
        # We add 3 colors. Max size len(s) + 3.
        # If s is empty (len 0), we need 3 distinct colors. K >= 3.
        # If s has size 1, we need 2 new distinct colors. K >= 3.
        # If s has size >= 2, we need 1 new color. K >= 3.
        return K >= 3

    if unassigned_count == 4:
        return K >= 3

    return True
