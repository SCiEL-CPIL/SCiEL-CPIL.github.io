def construct():
    import time, random
    from random import Random

    T0 = time.time()
    DEADLINE = 150.0
    MARGIN = 12.0

    def elapsed():
        return time.time() - T0

    def gen_graph(n, p, seed):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    adj[u].add(v); adj[v].add(u); edges.append((u, v))
        return adj, edges

    def is_clique(adj, verts):
        for i in range(len(verts)):
            a = verts[i]
            for j in range(i+1, len(verts)):
                if verts[j] not in adj[a]:
                    return False
        return True

    def greedy_pack(adj, edges, order):
        n = len(adj)
        uncovered = set(edges)
        cliques = []
        while uncovered:
            if elapsed() > DEADLINE - MARGIN:
                break
            # candidate degree
            cand_deg = [0]*n
            for (a,b) in uncovered:
                cand_deg[a]+=1; cand_deg[b]+=1
            best = None
            for start in order:
                if cand_deg[start]==0:
                    continue
                clique=[start]
                cand=[v for v in order if v!=start and v in adj[start]]
                i=0
                while i < len(cand):
                    v=cand[i]; ok=True
                    for c in clique:
                        if v not in adj[c]:
                            ok=False; break
                    if ok:
                        clique.append(v)
                        cand=[w for w in cand[i+1:] if all(w in adj[c] for c in clique)]
                        i=0
                    else:
                        i+=1
                if best is None or len(clique)>len(best):
                    best=clique
            if best is None or len(best)<2:
                for (a,b) in list(uncovered):
                    cliques.append([a,b])
                uncovered.clear()
                break
            cliques.append(best)
            for i in range(len(best)):
                a=best[i]
                for j in range(i+1,len(best)):
                    b=best[j]
                    uncovered.discard((a,b) if a<b else (b,a))
        return cliques

    def merge_refine(adj, cliques):
        changed=True
        while changed and elapsed() < DEADLINE - MARGIN:
            changed=False
            for i in range(len(cliques)):
                for j in range(i+1, len(cliques)):
                    if is_clique(adj, cliques[i]+cliques[j]):
                        cliques[i]=cliques[i]+cliques[j]
                        cliques.pop(j)
                        changed=True
                        break
                if changed:
                    break
        return cliques

    def complete(cliques, edges):
        covered=set()
        for c in cliques:
            for i in range(len(c)):
                a=c[i]
                for j in range(i+1,len(c)):
                    b=c[j]
                    covered.add((a,b) if a<b else (b,a))
        for e in edges:
            if e not in covered:
                cliques.append(list(e))
        return cliques

    def solve_one(adj, edges):
        n=len(adj)
        base_order=list(range(n))
        best=None
        # one deterministic
        c=greedy_pack(adj, edges, base_order)
        c=merge_refine(adj, c)
        c=complete(c, edges)
        best=c
        # randomized restarts
        rng=Random(12345)
        for _ in range(6):
            if elapsed() > DEADLINE - MARGIN:
                break
            order=base_order[:]
            rng.shuffle(order)
            c=greedy_pack(adj, edges, order)
            c=merge_refine(adj, c)
            c=complete(c, edges)
            if best is None or len(c)<len(best):
                best=c
        return best

    results=[]
    for (n,p,seed) in ((32,0.6,601),(64,0.55,602)):
        adj, edges = gen_graph(n,p,seed)
        results.append(solve_one(adj, edges))
    return results
