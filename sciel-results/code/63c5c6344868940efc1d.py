    if abs(x) < 0.15 and abs(y) < 0.15:
        row.append(0) # Air
    else:
        row.append(1) # Dielectric
    