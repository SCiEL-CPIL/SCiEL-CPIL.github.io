import time, math, random

def _triples(n):
    # all triples (x,y,z) with 1<=x<y<z<=n and (y-x)(z-x)==x*x
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
    
    # Precompute for each variable which clauses it belongs to and its position in each
    var_clauses = [[] for _ in range(n+1)]
    for ci, (a,b,c) in enumerate(T):
        var_clauses[a].append((ci, 0))
        var_clauses[b].append((ci, 1))
        var_clauses[c].append((ci, 2))
        
    # Helper to check if x can be added to S without creating violations
    def can_add(x, S):
        for (ci, pos) in var_clauses[x]:
            a, b, c = T[ci]
            others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
            o1, o2 = others[0]
            if S[o1] and S[o2]:
                return False
        return True

    # Structural Initialization:
    # Elements >= 90 cannot be the 'a' in any triple (since c > 2a > 180).
    # So they are only constrained as 'b' or 'c'.
    # Start with all [90, 180].
    S = [False]*(n+1)
    for i in range(90, n+1):
        S[i] = True
        
    # Add smaller elements greedily if they don't conflict with existing S
    # Process in increasing order? Or random?
    # Increasing order: small 'a's have fewer constraints on b,c?
    # Actually, we just check against current S.
    for a in range(1, 90):
        if can_add(a, S):
            S[a] = True
            
    best_S = S[:]
    best_size = sum(S)
    
    # Now use SA to improve
    random.seed(42)
    
    def violation_count(S):
        cnt = 0
        for (a,b,c) in T:
            if S[a] and S[b] and S[c]:
                cnt += 1
        return cnt

    def objective(S):
        size = sum(S)
        viol = violation_count(S)
        return size - 100 * viol

    cur_S = best_S[:]
    cur_obj = objective(cur_S)
    
    T_start = 5.0
    T_end = 0.1
    max_iter = 300000
    
    for it in range(max_iter):
        if time.time() - t0 > DEADLINE - 3:
            break
            
        x = random.randint(1, n)
        
        if cur_S[x]:
            delta_size = -1
            delta_viol = 0
            for (ci, pos) in var_clauses[x]:
                a, b, c = T[ci]
                others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
                o1, o2 = others[0]
                if cur_S[o1] and cur_S[o2]:
                    delta_viol -= 1
            delta_obj = delta_size - 100 * delta_viol
        else:
            delta_size = 1
            delta_viol = 0
            for (ci, pos) in var_clauses[x]:
                a, b, c = T[ci]
                others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
                o1, o2 = others[0]
                if cur_S[o1] and cur_S[o2]:
                    delta_viol += 1
            delta_obj = delta_size - 100 * delta_viol
            
        progress = it / max_iter
        T_cur = T_start * (T_end / T_start) ** progress
        
        if delta_obj >= 0 or random.random() < math.exp(delta_obj / T_cur):
            cur_S[x] = not cur_S[x]
            cur_obj += delta_obj
            
        if cur_obj > objective(best_S): # This is expensive, avoid
            pass # We need to track best_obj properly
            
    # Re-implement SA loop with proper best tracking to avoid calling objective() every step
    # The previous block was just a sketch. Let's write the real SA.
    
    cur_S = best_S[:]
    cur_obj = objective(cur_S)
    best_S = cur_S[:]
    best_obj = cur_obj
    
    for it in range(max_iter):
        if time.time() - t0 > DEADLINE - 3:
            break
            
        x = random.randint(1, n)
        
        if cur_S[x]:
            delta_size = -1
            delta_viol = 0
            for (ci, pos) in var_clauses[x]:
                a, b, c = T[ci]
                others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
                o1, o2 = others[0]
                if cur_S[o1] and cur_S[o2]:
                    delta_viol -= 1
            delta_obj = delta_size - 100 * delta_viol
        else:
            delta_size = 1
            delta_viol = 0
            for (ci, pos) in var_clauses[x]:
                a, b, c = T[ci]
                others = [(a,b) if pos==2 else ((a,c) if pos==1 else (b,c))]
                o1, o2 = others[0]
                if cur_S[o1] and cur_S[o2]:
                    delta_viol += 1
            delta_obj = delta_size - 100 * delta_viol
            
        progress = it / max_iter
        T_cur = T_start * (T_end / T_start) ** progress
        
        if delta_obj >= 0 or random.random() < math.exp(delta_obj / T_cur):
            cur_S[x] = not cur_S[x]
            cur_obj += delta_obj
            
        if cur_obj > best_obj:
            best_obj = cur_obj
            best_S = cur_S[:]
            
    # Final repair: ensure feasibility
    # If best_S has violations, remove elements from violated triples
    # Simple repair: iterate through triples, if violated, remove the element with highest "cost" (or random)
    # But we want to maximize size, so we should prefer keeping large sets.
    # The SA penalty should have kept it feasible.
    # Just to be safe:
    for (a,b,c) in T:
        if best_S[a] and best_S[b] and best_S[c]:
            # Remove one, say the smallest, to preserve larger numbers?
            # Or random?
            # Removing 'a' (smallest) might be better as it's in more triples?
            # Actually, removing any one breaks the triple.
            # Let's remove the one that is in the fewest other triples?
            # For now, just remove 'a'
            best_S[a] = False
            
    out = [i for i in range(1, n+1) if best_S[i]]
    if not out:
        out = [180]
    return out
