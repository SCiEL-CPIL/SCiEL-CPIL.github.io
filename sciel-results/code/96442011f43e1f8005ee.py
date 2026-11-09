import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 138
    n = 16
    
    edge_idx = {}
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            edge_list.append((i,j))
            idx += 1
    num_edges = len(edge_list)
    
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    k4s.append((a,b,c,d))
    num_k4s = len(k4s)
    
    k4_edges = []
    for (a,b,c,d) in k4s:
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        k4_edges.append([edge_idx[e] for e in edges])
    
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e in k4_edges[k4_idx]:
            edge_k4s[e].append(k4_idx)
    
    def k4_count(col, ki):
        es = k4_edges[ki]
        c0 = col[es[0]]; c1 = col[es[1]]; c2 = col[es[2]]
        c3 = col[es[3]]; c4 = col[es[4]]; c5 = col[es[5]]
        # Count distinct
        if c0==c1==c2==c3==c4==c5: return 1
        s = {c0,c1,c2,c3,c4,c5}
        return len(s)
    
    def all_cnts(col):
        return [k4_count(col, i) for i in range(num_k4s)]
    
    def viol(cnts):
        v = 0
        for c in cnts:
            if c < 5: v += 5-c
        return v
    
    base = [(i+j) % 17 for i in range(16) for j in range(i+1,16)]
    
    for k in [15,14,13]:
        if time.time() > deadline: break
        for rs in range(500):
            if time.time() > deadline: break
            mp = [random.randint(0,k-1) for _ in range(17)]
            col = [mp[c] for c in base]
            cnts = all_cnts(col)
            v = viol(cnts)
            if v == 0: return col
            T = 3.0
            for it in range(8000):
                if time.time() > deadline: break
                if v == 0: return col
                bad = [i for i,c in enumerate(cnts) if c<5]
                if not bad: return col
                ki = random.choice(bad)
                e = random.choice(k4_edges[ki])
                oc = col[e]
                nc = random.randint(0,k-1)
                if nc==oc: continue
                d = 0
                for k2 in edge_k4s[e]:
                    oc2 = cnts[k2]
                    col[e]=nc
                    nc2 = k4_count(col,k2)
                    col[e]=oc
                    d += (max(0,5-nc2)-max(0,5-oc2))
                if d<=0 or random.random()<2.71828**(-d/max(T,0.3)):
                    col[e]=nc
                    for k2 in edge_k4s[e]:
                        cnts[k2]=k4_count(col,k2)
                    v=viol(cnts)
                T*=0.9997
    return base
