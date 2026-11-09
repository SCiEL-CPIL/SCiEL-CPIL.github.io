import time, math, random

def _triples(n):
    T=[]
    for x in range(1,n+1):
        x2=x*x
        d=1
        while d*d<=x2:
            if x2%d==0:
                e=x2//d
                if d<e:
                    y=x+d; z=x+e
                    if y<=n and z<=n and y!=z:
                        T.append((x,y,z))
            d+=1
    return T

def construct():
    t0=time.time()
    DEADLINE=85.0
    n=180
    T=_triples(n)
    m=len(T)
    
    # Precompute: for each element, list of (triple_idx, other1, other2)
    elem_triples=[[] for _ in range(n+1)]
    for ti,(a,b,c) in enumerate(T):
        elem_triples[a].append((ti,b,c))
        elem_triples[b].append((ti,a,c))
        elem_triples[c].append((ti,a,b))
    
    # For fast conflict counting: for each element x, and each triple it's in,
    # store the other two elements
    # conflict_count(S, x) = sum over triples containing x of [S[o1] and S[o2]]
    
    def conflict_count(S, x):
        c=0
        for (ti,o1,o2) in elem_triples[x]:
            if S[o1] and S[o2]:
                c+=1
        return c
    
    def violated_triples(S):
        """Return list of triple indices that are violated (all 3 in S)"""
        vt=[]
        for ti,(a,b,c) in enumerate(T):
            if S[a] and S[b] and S[c]:
                vt.append(ti)
        return vt
    
    def feasible(S):
        for (a,b,c) in T:
            if S[a] and S[b] and S[c]:
                return False
        return True

    # Greedy construction: descending order
    S=[False]*(n+1)
    for x in range(n,0,-1):
        if conflict_count(S,x)==0:
            S[x]=True
    size=sum(S)
    
    # Simulated annealing phase
    def simulated_annealing(S, t0, deadline):
        S=S[:]
        best_S=S[:]
        best_size=sum(S)
        cur_size=sum(S)
        
        # Temperature schedule
        T_start=5.0
        T_end=0.01
        # We'll do iterations until deadline
        # Each iteration: pick a random element to flip
        
        # Precompute for faster delta calculation
        # When we flip x:
        #   If adding x (S[x] was False): new conflicts = number of triples where x,o1,o2 all in S after adding
        #   If removing x (S[x] was True): conflicts removed = number of triples where x,o1,o2 all in S before removing
        
        # Objective: maximize (size - penalty * violated)
        # We want 0 violated. Let's use: score = size - 1000 * num_violated
        # This way we strongly prefer feasibility
        
        PENALTY=1000
        
        def objective(S):
            sz=sum(S)
            nv=0
            for (a,b,c) in T:
                if S[a] and S[b] and S[c]:
                    nv+=1
            return sz - PENALTY*nv
        
        cur_obj=objective(S)
        best_obj=cur_obj
        
        # Random restarts within SA
        num_restarts=0
        max_restarts=50
        
        while time.time()-t0 < deadline - 3.0:
            # Run SA for a while
            T_cur=T_start
            iterations=0
            max_iter=200000
            no_improve=0
            
            while time.time()-t0 < deadline - 3.0 and iterations < max_iter:
                iterations+=1
                # Pick random element
                x=random.randint(1,n)
                
                # Compute delta in objective if we flip x
                if S[x]:
                    # Removing x: size decreases by 1
                    # Violated triples that contained x: after removal, those are no longer violated
                    # Count how many violated triples contain x
                    removed_viol=0
                    for (ti,o1,o2) in elem_triples[x]:
                        if S[o1] and S[o2]:
                            removed_viol+=1
                    delta_obj = -1 + PENALTY*removed_viol
                else:
                    # Adding x: size increases by 1
                    # New violated triples: triples containing x where other two are in S
                    new_viol=0
                    for (ti,o1,o2) in elem_triples[x]:
                        if S[o1] and S[o2]:
                            new_viol+=1
                    delta_obj = 1 - PENALTY*new_viol
                
                new_obj=cur_obj+delta_obj
                
                # Accept?
                if new_obj >= cur_obj:
                    accept=True
                else:
                    # Metropolis criterion
                    diff=cur_obj-new_obj
                    if T_cur>0:
                        r=random.random()
                        if r < math.exp(-diff/T_cur):
                            accept=True
                        else:
                            accept=False
                    else:
                        accept=False
                
                if accept:
                    S[x]=not S[x]
                    cur_obj=new_obj
                    if cur_obj>best_obj:
                        best_obj=cur_obj
                        best_S=S[:]
                        no_improve=0
                    else:
                        no_improve+=1
                else:
                    no_improve+=1
                
                # Cool down
                T_cur*=0.9995
                if T_cur<T_end:
                    T_cur=T_end
                
                if no_improve>5000:
                    break
            
            # Check if we found something better
            if best_obj>cur_obj:
                S=best_S[:]
                cur_obj=best_obj
            
            num_restarts+=1
            if num_restarts>=max_restarts:
                break
        
        return best_S, best_size
    
    # Run SA
    if time.time()-t0 < DEADLINE - 10.0:
        S_sa, size_sa = simulated_annealing(S, t0, DEADLINE)
        if feasible(S_sa) and size_sa > size:
            S=S_sa
            size=size_sa
        elif feasible(S_sa):
            # Even if not bigger, use it as starting point for further search
            S=S_sa
            size=size_sa
    
    # Final local improvement: try adding elements
    if time.time()-t0 < DEADLINE - 5.0:
        improved=True
        while improved and time.time()-t0 < DEADLINE - 5.0:
            improved=False
            for x in range(1,n+1):
                if not S[x]:
                    if conflict_count(S,x)==0:
                        S[x]=True
                        size+=1
                        improved=True
    
    out=[i for i in range(1,n+1) if S[i]]
    if not out:
        out=[180]
    return out
