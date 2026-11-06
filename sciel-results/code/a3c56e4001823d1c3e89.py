def construct():
    # GF(25) with x^2 + 2 = 0 mod 5
    # alpha^2 = 3
    
    def add(a, b):
        return ((a[0] + b[0]) % 5, (a[1] + b[1]) % 5)
        
    def sub(a, b):
        return ((a[0] - b[0]) % 5, (a[1] - b[1]) % 5)
        
    def mul(a, b):
        a0, a1 = a
        b0, b1 = b
        real = (a0*b0 + 3*a1*b1) % 5
        imag = (a0*b1 + a1*b0) % 5
        return (real, imag)
        
    # Generate elements
    elems = [(i, j) for i in range(5) for j in range(5)]
    
    # Find generator of order 24
    def get_order(e):
        cur = (1, 0)
        for k in range(1, 25):
            cur = mul(cur, e)
            if cur == (1, 0):
                return k
        return -1
        
    gen = None
    for e in elems:
        if e == (0, 0): continue
        if get_order(e) == 24:
            gen = e
            break
            
    if gen is None:
        # Should not happen
        return {'n': 1, 'colors': []}
        
    # Generate powers
    powers = []
    cur = (1, 0)
    for _ in range(24):
        powers.append(cur)
        cur = mul(cur, gen)
        
    # Map element to color (0, 1, 2)
    color_map = {}
    for k in range(24):
        color_map[powers[k]] = k % 3
        
    # Precompute colors for K25
    # colors_25[i][j] for i < j
    # We can just compute on the fly in the main loop to save memory/complexity,
    # but precomputing a 25x25 matrix is fine.
    c25 = [[0]*25 for _ in range(25)]
    for i in range(25):
        for j in range(i+1, 25):
            d = sub(elems[i], elems[j])
            c25[i][j] = color_map[d]
            c25[j][i] = color_map[d] # Symmetric
            
    # Construct K50
    colors = []
    for i in range(50):
        for j in range(i+1, 50):
            if i < 25 and j < 25:
                colors.append(c25[i][j])
            elif i >= 25 and j >= 25:
                colors.append(c25[i-25][j-25])
            else:
                colors.append(3)
                
    return {'n': 50, 'colors': colors}
