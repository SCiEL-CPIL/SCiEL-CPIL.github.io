u = S[rng.integers(len(S))]
nbrs = adj[u]
if len(nbrs)==0: 
    # u has no conflicts; removing u frees nothing new? Actually removing u decreases counts of nbrs (none). Can't help. pick another u
    continue
for _ in range(100):
    v = nbrs[rng.integers(len(nbrs))]
    if inS[v]: continue
    if count[v]==1:
        # valid swap
        ...
        break
