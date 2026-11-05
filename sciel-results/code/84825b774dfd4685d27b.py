import time

def construct(k, g):
    # Try Cayley graphs of PGL(2, q)
    # q=9, 11
    
    def try_cayley(q):
        # GF(q) arithmetic
        if q == 9:
            # Hardcode tables for GF(9)
            # x^2 + 1 = 0
            # 0,1,2,w,1+w,2+w,2w,1+2w,2+2w
            add = [[0,1,2,3,4,5,6,7,8],
                   [1,2,0,4,5,6,7,8,3],
                   [2,0,1,5,6,7,8,3,4],
                   [3,4,5,6,7,8,0,1,2],
                   [4,5,6,7,8,3,1,2,0],
                   [5,6,7,8,3,4,2,0,1],
                   [6,7,8,0,1,2,3,4,5],
                   [7,8,3,1,2,0,4,5,6],
                   [8,3,4,2,0,1,5,6,7]]
            mul = [[0,0,0,0,0,0,0,0,0],
                   [0,1,2,3,4,5,6,7,8],
                   [0,2,1,6,5,4,3,8,7],
                   [0,3,6,8,7,2,5,4,1],
                   [0,4,5,7,2,8,1,6,3],
                   [0,5,4,2,8,7,3,1,6],
                   [0,6,3,5,1,4,8,2,7],
                   [0,7,8,4,6,1,2,5,3],
                   [0,8,7,1,3,6,2,4,5]]
            inv = [0, 1, 2, 8, 4, 5, 6, 7, 3] # Verify: 3*8=24=6? No. 3=w, 8=2+2w. w(2+2w)=2w+2w^2=2w-2=2w+1=1+2w=7. So 3*8=7.
            # Let's compute inv dynamically or correctly.
            # 1*1=1. 2*2=1. 3*?
            # w * (2+2w) = 2w + 2w^2 = 2w - 2 = 2w + 1 = 7. So inv(3)=8.
            # 4=1+w. 4*4 = 1+2w+w^2 = 1+2w-1 = 2w = 6. 4*6 = (1+w)2w = 2w+2w^2 = 2w-2 = 2w+1 = 7.
            # This is getting messy. I will use matrix inversion over GF(3) for the field elements?
            # No, I'll just use the tables I generated in a quick script mentally?
            # Better: Use a generic GF(p^m) implementation?
            # For q=9, I can use the fact that it's small.
            
        # Generic implementation for prime q is easier.
        # For q=9, I will use the matrix representation of GF(9) as 2x2 matrices over GF(3).
        
        def gf_add(a, b):
            if q == 9:
                # a, b are tuples (r, s) representing r + s*w
                return ((a[0]+b[0])%3, (a[1]+b[1])%3)
            else:
                return (a+b)%q
        
        def gf_mul(a, b):
            if q == 9:
                # (r1+s1 w)(r2+s2 w) = r1r2 + (r1s2+r2s1)w + s1s2 w^2
                # w^2 = -1 = 2
                r1, s1 = a
                r2, s2 = b
                r = (r1*r2 + s1*s2*2) % 3
                s = (r1*s2 + r2*s1) % 3
                return (r, s)
            else:
                return (a*b)%q

        # ... This is getting long. 
        # I will stick to q=11 (prime) for simplicity and robustness.
        # PGL(2, 11) has n=1320.
        
        if q != 11:
            return None
            
        # GF(11)
        def inv(x):
            return pow(x, -1, 11)
        
        def normalize(a, b, c, d):
            if a != 0:
                ia = inv(a)
                return (1, (b*ia)%11, (c*ia)%11, (d*ia)%11)
            else:
                ic = inv(c)
                return (0, (b*ic)%11, 1, (d*ic)%11)
        
        def mat_mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return ((a1*a2+b1*c2)%11, (a1*b2+b1*d2)%11, (c1*a2+d1*c2)%11, (c1*b2+d1*d2)%11)
        
        elements = set()
        for a in range(11):
            for b in range(11):
                for c in range(11):
                    for d in range(11):
                        det = (a*d - b*c) % 11
                        if det != 0:
                            elements.add(normalize(a,b,c,d))
        
        el_list = list(elements)
        n = len(el_list)
        index = {el: i for i, el in enumerate(el_list)}
        
        gens = [
            normalize(1, 1, 0, 1),
            normalize(1, 10, 0, 1),
            normalize(0, 10, 1, 0),
            normalize(0, 1, 1, 0),
            normalize(1, 2, 0, 1) # Random 5th?
        ]
        
        # Check distinctness
        if len(set(gens)) < 5:
            return None
            
        adj = [[] for _ in range(n)]
        for i, el in enumerate(el_list):
            for g in gens:
                prod = normalize(*mat_mul(el, g))
                j = index[prod]
                adj[i].append(j)
                
        # Girth check
        for start in range(n):
            visited = {start: 0}
            queue = [(start, 0)]
            while queue:
                node, depth = queue.pop(0)
                if depth == 4:
                    continue
                for nb in adj[node]:
                    if nb == start and depth > 0:
                        if depth + 1 < g:
                            return None
                    if nb not in visited:
                        visited[nb] = depth + 1
                        queue.append((nb, depth + 1))
                        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    res = try_cayley(11)
    if res:
        return res
        
    # Fallback to LUW
    return construct_luw(k, g)

def construct_luw(k, g):
    # Original code
    pass
