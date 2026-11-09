if c0 == c1:
    if c2 == c3:
        if c0 == c2: v += 1 # 1 color
        else: v += 1 # 2 colors (c0, c2)
    else:
        # c0=c1, c2!=c3
        if c0 == c2 or c0 == c3: v += 1 # 2 colors
        else: pass # 3 colors (c0, c2, c3)
else:
    # c0 != c1
    if c2 == c3:
        if c0 == c2 or c1 == c2: v += 1 # 2 colors
        else: pass # 3 colors
    else:
        # c2 != c3
        if c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3:
             # Check if exactly 2
             # If any of c2, c3 matches c0 or c1, and c2!=c3, and c0!=c1
             # It's 2 colors if {c0,c1} intersects {c2,c3}
             if c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3:
                 # It could be 2 or 3.
                 # If c0==c2 and c1==c3 -> 2 colors (c0, c1)
                 # If c0==c2 and c1!=c3 -> 3 colors (c0, c1, c3)
                 # Let's just use the set for simplicity, it's fast enough.
                 pass
