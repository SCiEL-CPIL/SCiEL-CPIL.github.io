import time, random, math

def _precompute(n=180):
    # triples (x,y,z) 0-indexed, x<y<z, (y-x)*(z-x)==x^2
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
    # for each vertex, list of (other1,other2)
    pairs=[[] for _ in range(n)]
    for (i,j,k) in triples:
        pairs[i].append((j,k)); pairs[j].append((i,k)); pairs[k].append((i,j))
    return triples,pairs

def _count_violations(S, triples):
    v=0
    for (i,j,k) in triples:
        if S[i] and S[j] and S[k]: v+=1
    return v

def _greedy_add(S, pairs, n):
    """Try to add elements greedily. Returns modified S and size."""
    improved=True
    while improved:
        improved=False
        for v in range(n):
            if not S[v]:
                ok=True
                for (u,w) in pairs[v]:
                    if S[u] and S[w]:
                        ok=False; break
                if ok:
                    S[v]=1; improved=True
    return S, sum(S)

def _repair(S, triples, pairs, n):
    """Remove elements to fix violations. Returns modified S and size."""
    S = S[:]
    changed = True
    while changed:
        changed = False
        for (i,j,k) in triples:
            if S[i] and S[j] and S[k]:
                # remove the element with highest degree (most conflicts)
                di = len(pairs[i]); dj = len(pairs[j]); dk = len(pairs[k])
                if di >= dj and di >= dk:
                    S[i] = 0
                elif dj >= di and dj >= dk:
                    S[j] = 0
                else:
                    S[k] = 0
                changed = True
                break
    return S, sum(S)

def _two_swap_search(S, triples, pairs, n, deadline, max_iters=50000):
    """Try 2-swap moves: remove one element, add another. Can escape local optima."""
    cur_size = sum(S)
    cur_v = _count_violations(S, triples)
    improved = False
    iters = 0
    while iters < max_iters and time.time() < deadline:
        iters += 1
        # Find a violating triple
        found = False
        for (i,j,k) in triples:
            if S[i] and S[j] and S[k]:
                found = True
                break
        if not found:
            # No violations, try to add elements
            for v in range(n):
                if not S[v]:
                    ok = True
                    for (u,w) in pairs[v]:
                        if S[u] and S[w]:
                            ok = False; break
                    if ok:
                        S[v] = 1
                        cur_size += 1
                        improved = True
                        break
            if not improved:
                break
            continue
        
        # We have a violation (i,j,k). Try swapping one of them with another element.
        candidates = [i, j, k]
        random.shuffle(candidates)
        best_swap = None
        best_delta = 0
        
        for rem in candidates:
            if not S[rem]:
                continue
            # Try adding each non-member element
            for add in range(n):
                if add == rem or S[add]:
                    continue
                # Check if removing rem and adding add improves things
                # Count violations involving rem (will be removed)
                rem_viol = 0
                for (u,w) in pairs[rem]:
                    if S[u] and S[w]:
                        rem_viol += 1
                # Count new violations involving add (will be added)
                add_viol = 0
                for (u,w) in pairs[add]:
                    if S[u] and S[w]:
                        add_viol += 1
                delta = rem_viol - add_viol
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (rem, add)
        
        if best_swap and best_swap[1] != best_swap[0]:
            rem, add = best_swap
            S[rem] = 0
            S[add] = 1
            cur_size = sum(S)
            cur_v = _count_violations(S, triples)
            improved = True
        else:
            # Try random swap
            rem = random.choice([i,j,k])
            add = random.randrange(n)
            if add != rem and not S[add]:
                S[rem] = 0
                S[add] = 1
                cur_size = sum(S)
                cur_v = _count_violations(S, triples)
    
    return S, sum(S), _count_violations(S, triples)

def construct():
    t0=time.time()
    DEADLINE=85.0
    n=180
    triples,pairs=_precompute(n)
    random.seed(12345)
    
    best = [0]*n
    best_size = 0
    
    # Multiple restarts with different initial states
    num_restarts = 8
    restart_time = (DEADLINE - (time.time()-t0)) / num_restarts
    
    for restart in range(num_restarts):
        if time.time()-t0 > DEADLINE - 5.0:
            break
        
        # Different initial seeds for each restart
        if restart == 0:
            S=[0]*n
            for i in range(90,180):
                S[i]=1
        elif restart == 1:
            S=[0]*n
            for i in range(60,180):
                S[i]=1
        elif restart == 2:
            S=[0]*n
            for i in range(120,180):
                S[i]=1
        elif restart == 3:
            S=[0]*n
            random.seed(12345 + restart)
            for i in range(n):
                if random.random() < 0.5:
                    S[i]=1
        elif restart == 4:
            S=[0]*n
            random.seed(99999 + restart)
            for i in range(n):
                if random.random() < 0.4:
                    S[i]=1
        elif restart == 5:
            S=[0]*n
            # Start with only large numbers
            for i in range(130,180):
                S[i]=1
        elif restart == 6:
            S=[0]*n
            # Start with alternating pattern
            for i in range(n):
                if i % 3 != 0:
                    S[i]=1
        else:
            S=[0]*n
            random.seed(77777 + restart)
            for i in range(n):
                if random.random() < 0.35:
                    S[i]=1
        
        # Repair initial state
        S, cur_size = _repair(S, triples, pairs, n)
        cur_v = _count_violations(S, triples)
        
        # SA phase with 2-swap moves
        T=2.0
        sa_deadline = min(time.time()-t0 + restart_time * 0.7, DEADLINE - 5.0)
        while time.time()-t0 < sa_deadline:
            # Randomly choose between single flip and 2-swap
            if random.random() < 0.3:
                # 2-swap: remove one random element, add another
                v1 = random.randrange(n)
                v2 = random.randrange(n)
                if v1 == v2 or S[v1] == S[v2]:
                    continue
                # Compute delta
                if S[v1] == 1:  # removing v1, adding v2
                    rem_viol = 0
                    for (u,w) in pairs[v1]:
                        if S[u] and S[w]:
                            rem_viol += 1
                    add_viol = 0
                    for (u,w) in pairs[v2]:
                        if S[u] and S[w]:
                            add_viol += 1
                    dE = (rem_viol - add_viol) * 1000.0
                else:  # adding v1, removing v2
                    add_viol = 0
                    for (u,w) in pairs[v1]:
                        if S[u] and S[w]:
                            add_viol += 1
                    rem_viol = 0
                    for (u,w) in pairs[v2]:
                        if S[u] and S[w]:
                            rem_viol += 1
                    dE = (rem_viol - add_viol) * 1000.0
                
                if dE >= 0 or random.random() < math.exp(-dE/T):
                    S[v1] = 1 - S[v1]
                    S[v2] = 1 - S[v2]
                    cur_size = sum(S)
                    cur_v = _count_violations(S, triples)
                    if cur_v == 0 and cur_size > best_size:
                        best = S[:]; best_size = cur_size
            else:
                # Single flip
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
                new_score=new_size - 1000.0*new_v
                cur_score=cur_size - 1000.0*cur_v
                dE=new_score-cur_score
                if dE>=0 or random.random()<math.exp(dE/T):
                    S[v]=1-old
                    cur_v=new_v; cur_size=new_size
                    if new_v==0 and cur_size>best_size:
                        best=S[:]; best_size=cur_size
            T*=0.99995
            if T<0.05: T=0.05
        
        # 2-swap polish
        S, cur_size, cur_v = _two_swap_search(S, triples, pairs, n, min(time.time()+2.0, DEADLINE-5.0), max_iters=20000)
        if cur_v == 0 and cur_size > best_size:
            best = S[:]; best_size = cur_size
        
        # Greedy add from current S
        S, cur_size = _greedy_add(S, pairs, n)
        if cur_size > best_size:
            feasible = _count_violations(S, triples) == 0
            if feasible:
                best = S[:]; best_size = cur_size
        
        # Also try greedy from best
        Sb = best[:]
        Sb, sz = _greedy_add(Sb, pairs, n)
        if sz > best_size:
            feasible = _count_violations(Sb, triples) == 0
            if feasible:
                best = Sb; best_size = sz
    
    # Final polish: try multiple random orderings for greedy add
    for _ in range(5):
        if time.time() - t0 > DEADLINE - 2.0:
            break
        Sb = best[:]
        # Random permutation greedy add
        order = list(range(n))
        random.shuffle(order)
        improved = True
        while improved:
            improved = False
            for v in order:
                if not Sb[v]:
                    ok = True
                    for (u,w) in pairs[v]:
                        if Sb[u] and Sb[w]:
                            ok = False; break
                    if ok:
                        Sb[v] = 1
                        improved = True
        sz = sum(Sb)
        if sz > best_size and _count_violations(Sb, triples) == 0:
            best = Sb; best_size = sz
    
    out=[i+1 for i in range(n) if best[i]]
    if not out: out=[180]
    return out
