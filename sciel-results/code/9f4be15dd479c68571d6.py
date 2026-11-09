def construct():
    import time
    start_time = time.time()
    DEADLINE = 110  # safety margin
    
    # F4: 0,1,2(ω),3(ω²), ω²=ω+1
    F4_add = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
    F4_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    
    # PG(2,4) points
    pts, seen = [], set()
    inv = {0:0,1:1,2:3,3:2}
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x==0 and y==0 and z==0: continue
                if x: rep=(1, F4_mul(inv[x],y), F4_mul(inv[x],z))
                elif y: rep=(0,1, F4_mul(inv[y],z))
                else: rep=(0,0,1)
                if rep not in seen:
                    seen.add(rep); pts.append(rep)
    
    def dot(p,q):
        return F4_add[F4_add[F4_mul[p[0]][q[0]]][F4_mul[p[1]][q[1]]]][F4_mul[p[2]][q[2]]]
    
    n21 = len(pts)
    A = [[False]*n21 for _ in range(n21)]
    for i in range(n21):
        for j in range(i+1, n21):
            if dot(pts[i],pts[j])==0:
                A[i][j]=A[j][i]=True
    
    # Remove min-degree vertex
    dg = [sum(A[i]) for i in range(n21)]
    rm = min(range(n21), key=lambda i: dg[i])
    keep = [i for i in range(n21) if i!=rm]
    
    n = 20
    g = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if A[keep[i]][keep[j]]:
                g[i][j]=g[j][i]=True
    
    def can_add(g,i,j,n):
        for a in range(n):
            if a==i: continue
            for b in range(a+1,n):
                if b==i: continue
                if g[j][a] and g[j][b]:
                    c=0
                    for d in range(n):
                        if g[i][d] and g[a][d] and g[b][d]:
                            c+=1
                            if c>=2: return False
        for a in range(n):
            if a==j: continue
            for b in range(a+1,n):
                if b==j: continue
                if g[i][a] and g[i][b]:
                    c=0
                    for d in range(n):
                        if g[j][d] and g[a][d] and g[b][d]:
                            c+=1
                            if c>=2: return False
        return True
    
    # Greedy add
    imp=True
    while imp:
        imp=False
        for i in range(n):
            for j in range(i+1,n):
                if not g[i][j] and can_add(g,i,j,n):
                    g[i][j]=g[j][i]=True
                    imp=True
    
    # Edge swaps
    import random
    best = sum(sum(r) for r in g)//2
    for _ in range(30):
        if time.time()-start_time > DEADLINE: break
        el = [(i,j) for i in range(n) for j in range(i+1,n) if g[i][j]]
        if not el: break
        u,v = random.choice(el)
        g[u][v]=g[v][u]=False
        imp=True
        while imp:
            imp=False
            for i in range(n):
                for j in range(i+1,n):
                    if not g[i][j] and can_add(g,i,j,n):
                        g[i][j]=g[j][i]=True
                        imp=True
        ne = sum(sum(r) for r in g)//2
        if ne > best:
            best = ne
        else:
            g[u][v]=g[v][u]=True
            imp=True
            while imp:
                imp=False
                for i in range(n):
                    for j in range(i+1,n):
                        if not g[i][j] and can_add(g,i,j,n):
                            g[i][j]=g[j][i]=True
                            imp=True
    
    edges = [[i,j] for i in range(n) for j in range(i+1,n) if g[i][j]]
    return edges
