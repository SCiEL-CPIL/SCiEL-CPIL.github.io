def is_bad_fast(ap, colors):
    c0 = colors[ap[0]]
    c1 = colors[ap[1]]
    c2 = colors[ap[2]]
    c3 = colors[ap[3]]
    # Check if all same
    if c0 == c1 == c2 == c3:
        return True
    # Check if exactly 2 colors
    # This is equivalent to: the set of {c0,c1,c2,c3} has size 2
    # We can check:
    # (c0==c1 and c2==c3 and c0!=c2) OR
    # (c0==c2 and c1==c3 and c0!=c1) OR
    # (c0==c3 and c1==c2 and c0!=c1) OR
    # (c0==c1 and c0!=c2 and c2==c3) ... wait, this is getting complicated.
    # Simpler:
    distinct = 0
    if c1 != c0: distinct += 1
    if c2 != c0 and c2 != c1: distinct += 1
    if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
    return distinct < 2 # i.e. distinct is 0 or 1? No, distinct counts NEW colors.
    # If distinct < 2, it means we have at most 2 colors total?
    # If c0=c1=c2=c3, distinct=0. Total colors=1. Bad.
    # If c0=c1, c2=c3, c0!=c2. 
    #   c1!=c0 is False.
    #   c2!=c0 is True, c2!=c1 is False. distinct=1.
    #   c3!=c0 is False, c3!=c1 is False, c3!=c2 is False. distinct=1.
    #   Total colors=2. Bad.
    # If c0=c1=c2, c3!=c0.
    #   c1!=c0 False.
    #   c2!=c0 False.
    #   c3!=c0 True, c3!=c1 True, c3!=c2 True. distinct=1.
    #   Total colors=2. Bad.
    # If c0=c1, c2!=c0, c3!=c0, c3!=c2.
    #   c1!=c0 False.
    #   c2!=c0 True. distinct=1.
    #   c3!=c0 True, c3!=c1 True, c3!=c2 True. distinct=2.
    #   Total colors=3. Good.
    # So `distinct < 2` means total colors <= 2.
    return distinct < 2
