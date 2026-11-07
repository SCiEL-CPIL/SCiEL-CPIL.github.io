from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

trees = instance_trees()
total = 0
for k, t in enumerate(trees, 1):
    deg = [0]*k
    for a,b in t:
        deg[a]+=1; deg[b]+=1
    mx = max(deg)
    total += len(t)
    print(k, "edges:", len(t), "maxdeg:", mx)
print("TOTAL edges (max value):", total)
