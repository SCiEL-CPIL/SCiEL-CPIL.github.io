def construct():
    result = set(range(91, 181))
    
    def is_triple(a, b, c):
        # Check if a,b,c are distinct and form a reciprocal triple
        if a == b or b == c or a == c:
            return False
        # 1/a = 1/b + 1/c
        if b * c == a * (b + c):
            return True
        # 1/b = 1/a + 1/c
        if a * c == b * (a + c):
            return True
        # 1/c = 1/a + 1/b
        if a * b == c * (a + b):
            return True
        return False
    
    # Try to add numbers from 90 down to 1
    for a in range(90, 0, -1):
        if a in result:
            continue
        safe = True
        elems = list(result)
        for i in range(len(elems)):
            for j in range(i+1, len(elems)):
                b, c = elems[i], elems[j]
                if is_triple(a, b, c):
                    safe = False
                    break
            if not safe:
                break
        if safe:
            result.add(a)
    
    return list(result)
