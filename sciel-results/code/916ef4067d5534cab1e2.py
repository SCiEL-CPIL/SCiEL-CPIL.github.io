    for i in range(len(_DIVS)):
        d1 = _DIVS[i]
        if d1 > _TARGET: continue
        rem1 = _TARGET - d1
        for j in range(i+1, len(_DIVS)):
            d2 = _DIVS[j]
            if d2 > rem1: continue
            rem2 = rem1 - d2
            for k in range(j+1, len(_DIVS)):
                d3 = _DIVS[k]
                if d3 > rem2: continue
                rem3 = rem2 - d3
                if rem3 in _DIVS_SET:
                    if rem3 != d1 and rem3 != d2 and rem3 != d3:
                        return [d1, d2, d3, rem3]
