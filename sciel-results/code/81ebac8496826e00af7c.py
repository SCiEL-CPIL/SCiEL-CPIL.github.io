def construct():
    n = 512
    selected = set()
    
    for i in range(1, n + 1):
        valid = True
        # Check if i can be added without forming a 4-AP
        # The APs containing i are:
        # 1. i is the last term: (i-3d, i-2d, i-d, i)
        # 2. i is the 3rd term: (i-2d, i-d, i, i+d)
        # 3. i is the 2nd term: (i-d, i, i+d, i+2d)
        # 4. i is the 1st term: (i, i+d, i+2d, i+3d)
        
        # Case 1: i is the 4th term
        # d must be such that i-3d >= 1 => d <= (i-1)//3
        max_d1 = (i - 1) // 3
        for d in range(1, max_d1 + 1):
            if (i - d in selected) and (i - 2 * d in selected) and (i - 3 * d in selected):
                valid = False
                break
        if not valid:
            continue
            
        # Case 2: i is the 3rd term
        # d must be such that i+d <= n => d <= n-i
        max_d2 = n - i
        for d in range(1, max_d2 + 1):
            if (i - d in selected) and (i - 2 * d in selected) and (i + d in selected):
                valid = False
                break
        if not valid:
            continue
            
        # Case 3: i is the 2nd term
        # d must be such that i+2d <= n => d <= (n-i)//2
        max_d3 = (n - i) // 2
        for d in range(1, max_d3 + 1):
            if (i - d in selected) and (i + d in selected) and (i + 2 * d in selected):
                valid = False
                break
        if not valid:
            continue
            
        # Case 4: i is the 1st term
        # d must be such that i+3d <= n => d <= (n-i)//3
        max_d4 = (n - i) // 3
        for d in range(1, max_d4 + 1):
            if (i + d in selected) and (i + 2 * d in selected) and (i + 3 * d in selected):
                valid = False
                break
        if not valid:
            continue
            
        selected.add(i)
        
    return list(selected)
