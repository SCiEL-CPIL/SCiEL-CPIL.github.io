import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

def construct():
    n = 20
    # Indices for edges (i, j) with i < j
    edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m_edges = len(edge_pairs)
    edge_idx = {pair: k for k, pair in enumerate(edge_pairs)}
    
    # Variables:
    # 0 ... m_edges-1 : x_e for each edge e
    # m_edges ... m_edges + m_z - 1 : z_{uvw} for each triplet (u, v, w) with u < v, w not in {u, v}
    
    # Generate triplets (u, v, w) where u < v and w is not u or v
    # z_{uvw} represents x_{uw} * x_{vw}
    triplets = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w != u and w != v:
                    triplets.append((u, v, w))
    
    m_z = len(triplets)
    total_vars = m_edges + m_z
    
    # Objective: Maximize sum of x_e
    # milp minimizes c @ x, so we minimize -sum(x_e)
    c = np.zeros(total_vars)
    for k in range(m_edges):
        c[k] = -1.0
        
    # Bounds: All variables are binary [0, 1]
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    
    # Constraints
    # 1. Linearization constraints for each triplet (u, v, w)
    #    z_{uvw} <= x_{uw}
    #    z_{uvw} <= x_{vw}
    #    z_{uvw} >= x_{uw} + x_{vw} - 1  =>  -z_{uvw} + x_{uw} + x_{vw} >= 1  (Wait, x are binary)
    #    Actually: x_uw + x_vw - z_uvw <= 1 is equivalent to z >= x_uw + x_vw - 1?
    #    Let's check:
    #    If x_uw=1, x_vw=1, then z must be 1.
    #    Constraint: -z + x_uw + x_vw >= 1?
    #    If x_uw=1, x_vw=1, -z + 2 >= 1 => z <= 1. This doesn't force z=1.
    #    The correct linearization for z = x*y with x,y in {0,1} is:
    #    z <= x
    #    z <= y
    #    z >= x + y - 1
    #    z >= 0
    #    
    #    So:
    #    -z + x_uw >= 0  =>  -z + x_uw >= 0
    #    -z + x_vw >= 0  =>  -z + x_vw >= 0
    #    -z + x_uw + x_vw >= -1  =>  -z + x_uw + x_vw >= -1
    
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    # Add linearization constraints
    for idx, (u, v, w) in enumerate(triplets):
        z_var_idx = m_edges + idx
        # Need x_{uw} and x_{vw}. Ensure u < w and v < w for indexing?
        # edge_idx is defined for i < j.
        u1, u2 = sorted((u, w))
        v1, v2 = sorted((v, w))
        x_uw_idx = edge_idx[(u1, u2)]
        x_vw_idx = edge_idx[(v1, v2)]
        
        # 1. z <= x_uw  =>  z - x_uw <= 0
        row = np.zeros(total_vars)
        row[z_var_idx] = 1.0
        row[x_uw_idx] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # 2. z <= x_vw  =>  z - x_vw <= 0
        row = np.zeros(total_vars)
        row[z_var_idx] = 1.0
        row[x_vw_idx] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # 3. z >= x_uw + x_vw - 1  =>  -z + x_uw + x_vw >= -1
        row = np.zeros(total_vars)
        row[z_var_idx] = -1.0
        row[x_uw_idx] = 1.0
        row[x_vw_idx] = 1.0
        A_rows.append(row)
        lb_rows.append(-1.0)
        ub_rows.append(np.inf)

    # 2. K_{2,3} constraints: For each pair (u, v), sum of z_{uvw} over w != u, v <= 2
    #    We can group triplets by (u, v).
    #    Since we generated triplets in order of u, then v, then w, we can just iterate.
    #    Or we can create a mapping.
    
    # Let's just add a constraint for each pair (u, v)
    for u in range(n):
        for v in range(u + 1, n):
            row = np.zeros(total_vars)
            # Sum z_{uvw} for all w != u, v
            # Find indices of z variables for this (u, v)
            # We can iterate through triplets again or store indices.
            # For efficiency in construction, let's just loop through triplets and add to a dict or list.
            # Given small size, simple loop is fine.
            count = 0
            for idx, (tu, tv, tw) in enumerate(triplets):
                if tu == u and tv == v:
                    row[m_edges + idx] = 1.0
                    count += 1
            # We expect exactly n-2 = 18 variables for each (u, v)
            if count != 18:
                # This should not happen, but safety check
                pass
            
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(2.0)

    A = np.array(A_rows)
    lb = np.array(lb_rows)
    ub = np.array(ub_rows)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Solve
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(total_vars))
    
    if not res.success:
        # Fallback to seed if solver fails
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Extract edges
    x_vars = res.x[:m_edges]
    edges = []
    for k, (u, v) in enumerate(edge_pairs):
        if x_vars[k] > 0.5:
            edges.append([u, v])
            
    return edges
