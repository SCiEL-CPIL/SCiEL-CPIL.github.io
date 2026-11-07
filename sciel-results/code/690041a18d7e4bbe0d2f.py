def construct():
    limit = 256
    candidates = list(range(limit, 1, -1))
    result = []
    
    for x in candidates:
        # Check if x can be added to result
        valid = True
        # Check 1: x does not divide a*b for any distinct a,b in result
        for i in range(len(result)):
            a = result[i]
            for j in range(i + 1, len(result)):
                b = result[j]
                if (a * b) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
            
        # Check 2: a does not divide x*b for any distinct a,b in result
        for i in range(len(result)):
            a = result[i]
            for j in range(i + 1, len(result)):
                b = result[j]
                if (x * b) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
            
        # Check 3: b does not divide x*a for any distinct a,b in result
        # This is symmetric to Check 2, so it's redundant if we checked all pairs (a,b) in Check 2.
        # In Check 2, we iterated i, j over all distinct pairs.
        # So Check 3 is covered.
        
        result.append(x)
        
    return result
