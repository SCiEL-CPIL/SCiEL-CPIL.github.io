import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # 1. Map each edge to an index
    edge_to_idx = {}
    edges = []
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                edges.append((u, v))
                idx += 1
                
    # 2. Identify all 2-faces and their corresponding edge indices
    constraints_rows = []
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v3), max(v2, v3))
                e4 = (min(u, v2), max(u, v2))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                constraints_rows.append((idx1, idx2, idx3, idx4))

    n_constraints = len(constraints_rows)
    
    # 3. Setup ILP
    c = -np.ones(n_edges)
    
    rows = []
    cols = []
    data = []
    for k, (i1, i2, i3, i4) in enumerate(constraints_rows):
        rows.extend([k, k, k, k])
        cols.extend([i1, i2, i3, i4])
        data.extend([1, 1, 1, 1])
        
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_constraints, n_edges))
    
    lin_cons = LinearConstraint(A, ub=np.full(n_constraints, 3))
    
    integrality = np.ones(n_edges)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=bounds)
        if res.status == 0:
            x = np.round(res.x).astype(int)
            out = []
            for i in range(n_edges):
                if x[i] == 1:
                    out.append(list(edges[i]))
            return out
    except Exception:
        pass
        
    # Fallback: Greedy with random restarts if ILP fails
    import random
    best_edges = []
    for _ in range(10):
        random.shuffle(edges)
        chosen = set()
        out = []
        for u, v in edges:
            key = (u, v)
            ok = True
            # Check all faces containing this edge
            # An edge (u,v) is in faces defined by dim i (where u^v = 1<<i)
            # and any other dim j != i.
            # Face vertices: u, v, u^j, v^j (where j is the other dim)
            # Actually, let's just check the 4 edges of any face containing (u,v)
            # There are (d-1) faces containing a specific edge.
            diff = u ^ v
            i = diff.bit_length() - 1
            for j in range(d):
                if j == i: continue
                # Face is u, v, u^j, v^j
                # Edges: (u,v), (v, v^j), (u^j, v^j), (u, u^j)
                e1 = (min(u, v), max(u, v))
                e2 = (min(v, v^(1<<j)), max(v, v^(1<<j)))
                e3 = (min(u^(1<<j), v^(1<<j)), max(u^(1<<j), v^(1<<j)))
                e4 = (min(u, u^(1<<j)), max(u, u^(1<<j)))
                
                if e2 in chosen and e3 in chosen and e4 in chosen:
                    ok = False
                    break
            if ok:
                chosen.add(key)
                out.append([u, v])
        if len(out) > len(best_edges):
            best_edges = out
            
    return best_edges
