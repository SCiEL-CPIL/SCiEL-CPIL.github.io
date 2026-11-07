import time
import sys
from random import Random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

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

def construct():
    deadline = time.time() + 145
    trees = instance_trees(20, 611)
    n = 20
    
    # Precompute host edges
    host_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            host_edges.append((i, j))
    num_host_edges = len(host_edges) # 190
    
    # Precompute tree edges and vertices
    tree_vertices = []
    tree_edges = []
    for k in range(1, n + 1):
        tree_vertices.append(list(range(k)))
        tree_edges.append(trees[k-1])
        
    # Variables:
    # Y[k][i][j]: 1 if tree vertex i in tree k maps to host j
    # Z[k][e][u][v]: 1 if tree edge e in tree k maps to host edge (u,v)
    # Z1[k][e][u][v]: aux for (i->u, j->v)
    # Z2[k][e][u][v]: aux for (i->v, j->u)
    
    # We will flatten these into a single array for scipy.
    # Let's count variables.
    num_y = 0
    y_indices = {} # (k, i, j) -> index
    for k in range(n):
        for i in range(k+1):
            for j in range(n):
                y_indices[(k, i, j)] = num_y
                num_y += 1
                
    num_e = 0
    for k in range(n):
        num_e += len(tree_edges[k])
        
    num_z = num_e * num_host_edges
    z_indices = {} # (k, e_idx, he_idx) -> index
    z1_indices = {}
    z2_indices = {}
    
    idx = num_y
    for k in range(n):
        for e_idx in range(len(tree_edges[k])):
            for he_idx in range(num_host_edges):
                z_indices[(k, e_idx, he_idx)] = idx
                z1_indices[(k, e_idx, he_idx)] = idx + num_z
                z2_indices[(k, e_idx, he_idx)] = idx + 2 * num_z
                idx += 1
                
    total_vars = idx
    # total_vars ~ 4200 + 3 * 36100 = 112500
    
    # Objective: Maximize sum of Z (should be 190)
    c = np.zeros(total_vars)
    for k in range(n):
        for e_idx in range(len(tree_edges[k])):
            for he_idx in range(num_host_edges):
                c[z_indices[(k, e_idx, he_idx)]] = -1.0 # Minimize -sum Z

    # Constraints
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    # 1. Sum Y[k][i][j] = 1 for all k, i
    for k in range(n):
        for i in range(k+1):
            row = np.zeros(total_vars)
            for j in range(n):
                row[y_indices[(k, i, j)]] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)
            
    # 2. Sum Y[k][i][j] <= 1 for all k, j
    for k in range(n):
        for j in range(n):
            row = np.zeros(total_vars)
            for i in range(k+1):
                row[y_indices[(k, i, j)]] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
            
    # 3. Sum Z[k][e][he] = 1 for all k, e
    for k in range(n):
        for e_idx in range(len(tree_edges[k])):
            row = np.zeros(total_vars)
            for he_idx in range(num_host_edges):
                row[z_indices[(k, e_idx, he_idx)]] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)
            
    # 4. Sum Z[k][e][he] <= 1 for all he
    for he_idx in range(num_host_edges):
        row = np.zeros(total_vars)
        for k in range(n):
            for e_idx in range(len(tree_edges[k])):
                row[z_indices[(k, e_idx, he_idx)]] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
        
    # 5. Linking constraints
    # Z1 <= Y[i,u]
    # Z1 <= Y[j,v]
    # Z1 >= Y[i,u] + Y[j,v] - 1
    # Z2 <= Y[i,v]
    # Z2 <= Y[j,u]
    # Z2 >= Y[i,v] + Y[j,u] - 1
    # Z = Z1 + Z2
    
    for k in range(n):
        for e_idx in range(len(tree_edges[k])):
            i, j = tree_edges[k][e_idx]
            for he_idx in range(num_host_edges):
                u, v = host_edges[he_idx]
                
                z_idx = z_indices[(k, e_idx, he_idx)]
                z1_idx = z1_indices[(k, e_idx, he_idx)]
                z2_idx = z2_indices[(k, e_idx, he_idx)]
                
                y_iu = y_indices[(k, i, u)]
                y_iv = y_indices[(k, i, v)]
                y_ju = y_indices[(k, j, u)]
                y_jv = y_indices[(k, j, v)]
                
                # Z = Z1 + Z2 => Z - Z1 - Z2 = 0
                row = np.zeros(total_vars)
                row[z_idx] = 1.0
                row[z1_idx] = -1.0
                row[z2_idx] = -1.0
                A_eq.append(row)
                b_eq.append(0.0)
                
                # Z1 <= Y_iu
                row = np.zeros(total_vars)
                row[z1_idx] = 1.0
                row[y_iu] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
                # Z1 <= Y_jv
                row = np.zeros(total_vars)
                row[z1_idx] = 1.0
                row[y_jv] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
                # Z1 >= Y_iu + Y_jv - 1 => -Z1 + Y_iu + Y_jv <= 1
                row = np.zeros(total_vars)
                row[z1_idx] = -1.0
                row[y_iu] = 1.0
                row[y_jv] = 1.0
                A_ub.append(row)
                b_ub.append(1.0)
                
                # Z2 <= Y_iv
                row = np.zeros(total_vars)
                row[z2_idx] = 1.0
                row[y_iv] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
                # Z2 <= Y_ju
                row = np.zeros(total_vars)
                row[z2_idx] = 1.0
                row[y_ju] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
                # Z2 >= Y_iv + Y_ju - 1 => -Z2 + Y_iv + Y_ju <= 1
                row = np.zeros(total_vars)
                row[z2_idx] = -1.0
                row[y_iv] = 1.0
                row[y_ju] = 1.0
                A_ub.append(row)
                b_ub.append(1.0)

    # Convert to arrays
    A_ub_arr = np.array(A_ub)
    b_ub_arr = np.array(b_ub)
    A_eq_arr = np.array(A_eq)
    b_eq_arr = np.array(b_eq)
    
    constraints = [
        LinearConstraint(A_ub_arr, -np.inf, b_ub_arr),
        LinearConstraint(A_eq_arr, b_eq_arr, b_eq_arr)
    ]
    
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(total_vars))
        if res.status == 0:
            # Extract solution
            x = res.x
            result = [None] * 20
            for k in range(n):
                mapping = [0] * (k+1)
                for i in range(k+1):
                    for j in range(n):
                        if x[y_indices[(k, i, j)]] > 0.5:
                            mapping[i] = j
                            break
                result[k] = mapping
            return result
    except Exception as e:
        pass
        
    # Fallback to backtracking if ILP fails or times out
    return _backtrack_fallback(trees, n, deadline)

def _backtrack_fallback(trees, n, deadline):
    # Simple backtracking as before
    all_edges = set((i, j) for i in range(n) for j in range(i+1, n))
    available = set(all_edges)
    result = [None] * 20
    
    tree_order = list(range(20))
    tree_order.sort(key=lambda idx: -(idx + 1))
    
    def try_embed(t_idx, available, rng):
        k = t_idx + 1
        if k == 1:
            return [rng.randint(0, n - 1)]
        if k == 2:
            for u, v in available:
                return [u, v]
            return None
        tree_edges = trees[t_idx]
        tree_adj = {v: [] for v in range(k)}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        ordered_verts = sorted(range(k), key=lambda v: (-len(tree_adj[v]), v))
        host_map = {}
        used_hosts = set()
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available:
                        return False
                return True
            tv = ordered_verts[pos]
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                ok = True
                for nb in tree_adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in available:
                            ok = False
                            break
                if ok:
                    candidates.append(h)
            if not candidates:
                return False
            rng.shuffle(candidates)
            for h in candidates:
                host_map[tv] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_hosts.remove(h)
            return False
        
        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    for attempt in range(1000):
        if time.time() > deadline:
            break
        rng = Random(attempt * 31 + 7)
        avail = set(all_edges)
        res = [None] * 20
        success = True
        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            k = t_idx + 1
            if k == 1:
                res[t_idx] = [rng.randint(0, n - 1)]
                continue
            embedded = None
            for trial in range(5):
                if time.time() > deadline:
                    success = False
                    break
                embedded = try_embed(t_idx, avail, rng)
                if embedded is not None:
                    break
            if not success:
                break
            if embedded is None:
                success = False
                break
            res[t_idx] = embedded
            if k >= 2:
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    avail.discard(edge)
        if success and len(avail) == 0:
            return res
    return [None] * 19 + [list(range(20))]

