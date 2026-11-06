def construct():
    n = 128
    # Try increasing number of colors
    for K in range(1, 129):
        # Try to solve with colors 0..K-1
        colors = [-1] * n
        colors[0] = 0
        success = True
        
        for i in range(1, n):
            forbidden = 0
            # 4-term progressions ending at i
            for d in range(1, i // 3 + 1):
                A = colors[i - d]
                B = colors[i - 2 * d]
                D = colors[i - 3 * d]
                
                if A == B and B == D:
                    # Dead end, should not happen if 3-term constraint is enforced
                    success = False
                    break
                
                if A == B or A == D or B == D:
                    # Exactly 2 distinct
                    if A == B:
                        v1, v2 = A, D
                    elif A == D:
                        v1, v2 = A, B
                    else:
                        v1, v2 = B, A
                    forbidden |= (1 << v1)
                    forbidden |= (1 << v2)
            
            if not success:
                break
                
            # 3-term conditions
            for d in range(1, i // 2 + 1):
                A = colors[i - d]
                B = colors[i - 2 * d]
                if A == B:
                    forbidden |= (1 << A)
            
            # Find smallest available color in 0..K-1
            found = False
            for c in range(K):
                if not (forbidden & (1 << c)):
                    colors[i] = c
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            return colors
            
    # Fallback (should not happen)
    return [0] * n
