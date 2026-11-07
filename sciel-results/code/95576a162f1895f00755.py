import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    submasks = [[] for _ in range(N)]
    supermasks = [[] for _ in range(N)]
    
    for z in range(N):
        sub = z
        while True:
            submasks[z].append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & z
        complement = (~z) & (N - 1)
        s = complement
        while True:
            supermasks[z].append(z | s)
            if s == 0:
                break
            s = (s - 1) & complement
    
    def is_comp(a, b):
        return (a & b) == a or (a & b) == b
    
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_comp(family[i], family[j]):
                    count += 1
        return count
    
    def build_comp_in_F(family):
        comp_in_F = [0] * N
        for z in family:
            for y in submasks[z]:
                comp_in_F[y] += 1
            for y in supermasks[z]:
                comp_in_F[y] += 1
        fam_set = set(family)
        for y in fam_set:
            comp_in_F[y] -= 1
        return comp_in_F
    
    def local_search(family, max_iters, deadline):
        comp_in_F = build_comp_in_F(family)
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        fam_set = set(family)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            for i in range(m):
                x = family[i]
                k_x = comp_in_F[x] - 1
                
                best_y = -1
                best_val = -1
                
                for y in range(N):
                    if y in fam_set:
                        continue
                    val = comp_in_F[y]
                    if is_comp(y, x):
                        val -= 1
                    if val > best_val:
                        best_val = val
                        best_y = y
                
                if best_y == -1 or best_val <= k_x:
                    continue
                
                for y in submasks[x]:
                    comp_in_F[y] -= 1
                for y in supermasks[x]:
                    comp_in_F[y] -= 1
                comp_in_F[x] += 1
                
                for y in submasks[best_y]:
                    comp_in_F[y] += 1
                for y in supermasks[best_y]:
                    comp_in_F[y] += 1
                comp_in_F[best_y] -= 1
                
                fam_set.remove(x)
                fam_set.add(best_y)
                family[i] = best_y
                current_score += best_val - k_x
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
                
                improved = True
            
            if not improved:
                break
                
        return best_family, best_score
    
    best_overall = None
    best_score_overall = 0
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        random.seed(42 + restart)
        family = random.sample(range(N), m)
        
        fam, score = local_search(family, 200, deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam
    
    if time.time() < deadline - 15:
        chain = [0]
        cur = 0
        for b in range(n):
            cur |= (1 << b)
            chain.append(cur)
        remaining = [x for x in range(N) if x not in chain]
        random.seed(999)
        extra = random.sample(remaining, m - len(chain))
        family = chain + extra
        fam, score = local_search(family, 200, deadline)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = fam
    
    if best_overall is not None and time.time() < deadline - 5:
        best_overall, best_score_overall = local_search(best_overall, 500, deadline)
    
    if best_overall is None:
        best_overall = list(range(m))
    
    result = []
    for mask in best_overall:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
