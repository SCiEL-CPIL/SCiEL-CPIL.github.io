    ring2_axial = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            dist = (abs(q) + abs(r) + abs(q+r)) / 2
            if dist == 2:
                ring2_axial.append((q, r))
