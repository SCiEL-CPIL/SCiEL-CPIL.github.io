"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Good starting point: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def total_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    best_cost = total_cost(p)
    best_p = p.copy()
    
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Local search with 2-opt swaps
    # Precompute for efficiency
    while time.time() < deadline:
        improved = False
        # Try all pairs
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Compute delta for swapping p[i] and p[j]
                x, y = p[i], p[j]
                # Change in cost
                # Terms where i is first index
                delta = np.dot(a[i, :], b[y, p] - b[x, p])
                # Terms where j is first index
                delta += np.dot(a[j, :], b[x, p] - b[y, p])
                # Terms where i is second index
                delta += np.dot(a[:, i], b[p, y] - b[p, x])
                # Terms where j is second index
                delta += np.dot(a[:, j], b[p, x] - b[p, y])
                # Correct for double-counted terms (i,i), (i,j), (j,i), (j,j)
                # (i,i): counted in first and third sums
                # In first sum (k=i): a[i,i]*(b[y,x]-b[x,x])
                # In third sum (k=i): a[i,i]*(b[x,y]-b[x,x])
                # Actual change: a[i,i]*(b[y,y]-b[x,x])
                # Overcount: a[i,i]*(b[y,x]+b[x,y]-2*b[x,x]) - a[i,i]*(b[y,y]-b[x,x])
                # = a[i,i]*(b[y,x]+b[x,y]-b[y,y]-b[x,x])
                delta -= a[i,i] * (b[y,x] + b[x,y] - b[y,y] - b[x,x])
                # (j,j): similar
                delta -= a[j,j] * (b[x,y] + b[y,x] - b[x,x] - b[y,y])
                # (i,j): counted in first sum (k=j) and fourth sum (k=i)
                # First sum (k=j): a[i,j]*(b[y,y]-b[x,y])
                # Fourth sum (k=i): a[i,j]*(b[x,x]-b[x,y])  wait, b[p[i],x] where p[i]=x, so b[x,x]
                # Actually fourth sum is a[k,j]*(b[p[k],x]-b[p[k],y]), for k=i: a[i,j]*(b[x,x]-b[x,y])
                # Actual change for (i,j): a[i,j]*(b[y,x]-b[x,y])
                # Overcount: a[i,j]*(b[y,y]-b[x,y] + b[x,x]-b[x,y]) - a[i,j]*(b[y,x]-b[x,y])
                # = a[i,j]*(b[y,y]+b[x,x]-2*b[x,y]-b[y,x]+b[x,y])
                # = a[i,j]*(b[y,y]+b[x,x]-b[x,y]-b[y,x])
                delta -= a[i,j] * (b[y,y] + b[x,x] - b[x,y] - b[y,x])
                # (j,i): counted in second sum (k=i) and third sum (k=j)
                # Second sum (k=i): a[j,i]*(b[x,x]-b[y,x])
                # Third sum (k=j): a[j,i]*(b[y,y]-b[y,x])  wait, b[p[j],y] where p[j]=y, so b[y,y]
                # Actually third sum is a[k,i]*(b[p[k],y]-b[p[k],x]), for k=j: a[j,i]*(b[y,y]-b[y,x])
                # Actual change for (j,i): a[j,i]*(b[x,y]-b[y,x])
                # Overcount: a[j,i]*(b[x,x]-b[y,x] + b[y,y]-b[y,x]) - a[j,i]*(b[x,y]-b[y,x])
                # = a[j,i]*(b[x,x]+b[y,y]-2*b[y,x]-b[x,y]+b[y,x])
                # = a[j,i]*(b[x,x]+b[y,y]-b[y,x]-b[x,y])
                delta -= a[j,i] * (b[x,x] + b[y,y] - b[y,x] - b[x,y])
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    improved = True
        if not improved:
            break
    
    if int(total_cost(best_p)) < best_cost:
        pass  # best_p is already tracked
    
    return p.tolist()
# EVOLVE-BLOCK-END
