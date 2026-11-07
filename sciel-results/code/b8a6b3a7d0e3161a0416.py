"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def solve_k(k):
        assignment = [-1] * n
        
        # active_edges[c1][c2] = list of (u,v) edges where both endpoints assigned and have colors c1,c2
        # We'll use a dict of dicts
        active = {}
        
        def add_edge_to_pair(u, v, c1, c2):
            key = (min(c1,c2), max(c1,c2))
            if key not in active:
                active[key] = []
            active[key].append((u,v))
        
        def remove_edge_from_pair(u, v, c1, c2):
            key = (min(c1,c2), max(c1,c2))
            if key in active:
                active[key].remove((u,v))
                if not active[key]:
                    del active[key]
        
        def has_cycle(c1, c2):
            key = (min(c1,c2), max(c1,c2))
            if key not in active:
                return False
            elist = active[key]
            # Quick check: need at least as many edges as vertices for a cycle
            verts = set()
            for u,v in elist:
                verts.add(u)
                verts.add(v)
            if len(elist) < len(verts):
                return False
            
            # Union-find
            idx = {}
            for i,v in enumerate(verts):
                idx[v] = i
            m = len(verts)
            parent = list(range(m))
            rank = [0]*m
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x,y):
                rx,ry = find(x),find(y)
                if rx==ry: return False
                if rank[rx]<rank[ry]: rx,ry=ry,rx
                parent[ry]=rx
                if rank[rx]==rank[ry]: rank[rx]+=1
                return True
            
            for u,v in elist:
                if not union(idx[u],idx[v]):
                    return True
            return False
        
        def assign(v, c):
            assignment[v] = c
            for nb in adj[v]:
                if assignment[nb] != -1:
                    cn = assignment[nb]
                    if cn != c:
                        add_edge_to_pair(v, nb, c, cn)
        
        def unassign(v):
            c = assignment[v]
            for nb in adj[v]:
                if assignment[nb] != -1:
                    cn = assignment[nb]
                    if cn != c:
                        remove_edge_from_pair(v, nb, c, cn)
            assignment[v] = -1
        
        def backtrack():
            min_dom = float('inf')
            min_v = -1
            min_cols = None
            
            for i in range(n):
                if assignment[i] == -1:
                    used = set()
                    for nb in adj[i]:
                        if assignment[nb] != -1:
                            used.add(assignment[nb])
                    avail = [c for c in range(k) if c not in used]
                    if len(avail) < min_dom:
                        min_dom = len(avail)
                        min_v = i
                        min_cols = avail
                        if min_dom == 0:
                            return False
            
            if min_v == -1:
                return True
            
            for c in min_cols:
                assign(min_v, c)
                
                ok = True
                for other in range(k):
                    if other == c:
                        continue
                    if has_cycle(c, other):
                        ok = False
                        break
                
                if not ok:
                    unassign(min_v)
                    continue
                
                if backtrack():
                    return True
                
                unassign(min_v)
            
            return False
        
        assign(0, 0)
        if backtrack():
            return list(assignment)
        unassign(0)
        return None
    
    for k in [3, 4, 5, 6]:
        res = solve_k(k)
        if res is not None:
            for i in range(n):
                assert isinstance(res[i], int)
                assert 0 <= res[i] < 50
            return res
    
    return list(range(n))
# EVOLVE-BLOCK-END
