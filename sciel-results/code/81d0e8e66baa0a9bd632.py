def construct():
    # Base set: 63 to 128
    s = set(range(63, 129))
    
    # Try to add 62
    # If we add 62, we must ensure no forbidden triples.
    # We identified that adding 62 to {63..128} creates issues with 63,64 and 63,65.
    # Removing 125 breaks both.
    s.add(62)
    s.remove(125)
    
    # Verify validity
    lst = list(s)
    valid = True
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            for k in range(j+1, len(lst)):
                a, b, c = lst[i], lst[j], lst[k]
                if a+b in s and a+c in s and b+c in s:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to safe set
        return list(range(63, 129))
        
    return lst
