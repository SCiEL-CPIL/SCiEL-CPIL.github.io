def _color16():
    n = 16
    colors = []
    for a in range(2): # 2
        for b in range(4): # 4
            for a2 in range(2): # 2
                for b2 in range(4): # 4
                    if a == a2 and b == b2:
                        continue
