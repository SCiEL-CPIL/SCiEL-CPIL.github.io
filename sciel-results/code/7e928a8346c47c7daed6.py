import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 285  # 15s margin
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
                
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    def find_4cliques(adj, n):
        cliques = []
        for i in range(n):
            for j in range(i+1, n):
                if j not in adj[i]:
                    continue
                for k in range(j+1, n):
                    if k not in adj[i] or k not in adj[j]:
                        continue
                    for l in range(k+1, n):
                        if l not in adj[i] or l not in adj[j] or l not in adj[k]:
                            continue
                        cliques.append((i, j, k, l))
        return cliques
    
    def find_4independent(adj, n):
        independent = []
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    continue
                for k in range(j+1, n):
                    if k in adj[i] or k in adj[j]:
                        continue
                    for l in range(k+1, n):
                        if l in adj[i] or l in adj[j] or l in adj[k]:
                            continue
                        independent.append((i, j, k, l))
        return independent
    
    target_n = 42
    current_adj = [s.copy() for s in best_adj]
    current_n = best_n
    
    def try_add_vertex(adj, n, four_cliques, four_independent, max_iters=10000):
        S = set()
        for i in range(n):
            if random.random() < 0.5:
                S.add(i)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            violations = []
            for clique in four_cliques:
                if all(v in S for v in clique):
                    violations.append(('clique', clique))
            for indep in four_independent:
                if all(v not in S for v in indep):
                    violations.append(('indep', indep))
            
            if not violations:
                return S
            
            vtype, vset = random.choice(violations)
            if vtype == 'clique':
                v = random.choice(vset)
                S.discard(v)
            else:
                v = random.choice(vset)
                S.add(v)
        
        return None
    
    for new_v in range(current_n, target_n):
        if time.time() > deadline:
            break
            
        four_cliques = find_4cliques(current_adj, current_n)
        four_independent = find_4independent(current_adj, current_n)
        
        S = try_add_vertex(current_adj, current_n, four_cliques, four_independent)
        
        if S is None:
            break
        
        current_adj.append(S)
        for i in S:
            current_adj[i].add(current_n)
        current_n += 1
        
        best_n = current_n
        best_adj = [s.copy() for s in current_adj]
        
        if time.time() > deadline:
            break
    
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append((i, j))
                
    return best_n, edges
