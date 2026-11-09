import time, random, math

def _precompute(n=180):
    triples=[]
    for x in range(1,n+1):
        x2=x*x
        d=1
        while d*d<=x2:
            if x2%d==0:
                e=x2//d
                if d<e:
                    y=x+d
                    z=x+e
                    if y<=n and z<=n and y!=z:
                        triples.append((x-1,y-1,z-1))
            d+=1
    pairs=[[] for _ in range(n)]
    for (i,j,k) in triples:
        pairs[i].append((j,k)); pairs[j].append((i,k)); pairs[k].append((i,j))
    return triples,pairs

def construct():
    t0=time.time()
    DEADLINE=85.0
    n=180
    triples,pairs=_precompute(n)
    random.seed(12345)
    
    def greedy_add(S, order):
        sz=sum(S)
        improved=True
        while improved:
            improved=False
            for v in order:
                if not S[v]:
                    ok=True
                    for (u,w) in pairs[v]:
                        if S[u] and S[w]:
                            ok=False; break
                    if ok:
                        S[v]=1; sz+=1; improved=True
        return S, sz
    
    def sa_run(S, T_init, T_min, cool, deadline):
        cur_v=0
        for (i,j,k) in triples:
            if S[i] and S[j] and S[k]: cur_v+=1
        cur_size=sum(S)
        best=S[:]; best_size=cur_size
        T=T_init
        while time.time()<deadline:
            v=random.randrange(n)
            old=S[v]
            if old==0:
                addc=0
                for (u,w) in pairs[v]:
                    if S[u] and S[w]: addc+=1
                new_v=cur_v+addc
                new_size=cur_size+1
            else:
                remc=0
                for (u,w) in pairs[v]:
                    if S[u] and S[w]: remc+=1
                new_v=cur_v-remc
                new_size=cur_size-1
            new_score=new_size - 500.0*new_v
            cur_score=cur_size - 500.0*cur_v
            dE=new_score-cur_score
            if dE>=0 or random.random()<math.exp(dE/T):
                S[v]=1-old
                cur_v=new_v; cur_size=new_size
                if new_v==0 and cur_size>best_size:
                    best=S[:]; best_size=cur_size
            T*=cool
            if T<T_min: T=T_min
        return best, best_size
    
    # Initial: high numbers
    S=[0]*n
    for i in range(89,180):
        S[i]=1
    S, sz = greedy_add(S, range(n))
    best=S[:]; best_size=sz
    
    # SA from current best
    remaining=DEADLINE-time.time()
    if remaining>5:
        S2=best[:]
        sa_best, sa_size = sa_run(S2, 1.5, 0.01, 0.99998, t0+remaining-2)
        if sa_size>best_size:
            best=sa_best; best_size=sa_size
        S, sz = greedy_add(best, range(n))
        if sz>best_size:
            best_size=sz
    
    # Try different initial seeds
    for seed in [42, 7, 999]:
        if time.time()>DEADLINE-5: break
        random.seed(seed)
        S3=[0]*n
        for i in range(n):
            if random.random()<0.55:
                S3[i]=1
        S3, sz3 = greedy_add(S3, range(n))
        if sz3>best_size:
            best=S3[:]; best_size=sz3
        remaining=DEADLINE-time.time()
        if remaining>5:
            S4=best[:]
            sa_best4, sa_size4 = sa_run(S4, 1.0, 0.01, 0.99997, t0+remaining-2)
            if sa_size4>best_size:
                best=sa_best4; best_size=sa_size4
            S, sz = greedy_add(best, range(n))
            if sz>best_size:
                best_size=sz
    
    # Final greedy polish from best
    S_final=best[:]
    S_final, final_size = greedy_add(S_final, range(n))
    if final_size>best_size:
        best=S_final; best_size=final_size
    
    # Also try reverse order greedy
    S_rev=best[:]
    S_rev, rev_size = greedy_add(S_rev, range(n-1,-1,-1))
    if rev_size>best_size:
        best=S_rev; best_size=rev_size
    
    out=[i+1 for i in range(n) if best[i]]
    if not out: out=[180]
    return out
