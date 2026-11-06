import time
import sys
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Build Conflict Graph
    # adj[i] is a bitmask of conflicts for i (0-indexed, representing i+1)
    adj = [0] * N
    
    for u in range(N):
        val_u = u + 1
        for v in range(u + 1, N):
            val_v = v + 1
            # Case 1: u, v are ends. Middle m = (val_u + val_v) / 2
            if (val_u + val_v) % 2 == 0:
                m_val = (val_u + val_v) // 2
                if 1 <= m_val <= N:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Case 2: u is middle, v is end. Third w = 2*val_u - val_v
            w_val = 2 * val_u - val_v
            if 1 <= w_val <= N and w_val != val_u and w_val != val_v:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Case 3: v is middle, u is end. Third w = 2*val_v - val_u
            w_val = 2 * val_v - val_u
            if 1 <= w_val <= N and w_val != val_u and w_val != val_v:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # Weights: w[i] = 1 / (i+1)
    weights = np.array([1.0 / (i + 1) for i in range(N)])

    # 2. Formulate ILP
    # Variables: x_i in {0, 1} for i in 0..N-1
    # Objective: Maximize sum(weights[i] * x_i)
    # Constraints: x_u + x_v <= 1 for all (u, v) such that adj[u] has bit v set.
    
    # Collect constraints
    # We need to build the constraint matrix A_ub and b_ub.
    # A_ub will be a sparse matrix or dense. N=256, edges ~ N^2/2 potentially dense.
    # Let's count edges first to decide on density.
    
    edge_list = []
    for u in range(N):
        mask = adj[u]
        # Only consider v > u to avoid duplicates
        mask &= ~((1 << (u + 1)) - 1) # Mask out bits <= u
        v = u + 1
        while mask:
            if mask & 1:
                edge_list.append((u, v))
            mask >>= 1
            v += 1
            
    # Actually, the bit shifting loop above is slightly off because mask bits correspond to absolute indices.
    # Let's just iterate v > u properly.
    edge_list = []
    for u in range(N):
        for v in range(u + 1, N):
            if (adj[u] >> v) & 1:
                edge_list.append((u, v))
                
    num_constraints = len(edge_list)
    
    # For N=256, the graph of 3-AP conflicts is quite dense.
    # Max edges: Each pair can conflict. Total pairs ~ 32k.
    # A dense 32k x 256 matrix is ~ 8M floats. 64MB. This is acceptable.
    # However, HiGHS prefers sparse. Let's use scipy.sparse.
    
    from scipy.sparse import csr_matrix
    
    rows = []
    cols = []
    data = []
    
    for u, v in edge_list:
        rows.append(len(data))
        cols.append(u)
        data.append(1.0)
        rows.append(len(data))
        cols.append(v)
        data.append(1.0)
        
    A_ub = csr_matrix((data, (rows, cols)), shape=(num_constraints, N))
    b_ub = np.ones(num_constraints)
    
    # Objective: minimize -weights . x
    c = -weights
    
    # Bounds: x_i in [0, 1]
    integrality = np.ones(N)
    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
    
    # Constraints: A_ub @ x <= b_ub
    # In scipy.optimize.milp, constraints are defined as:
    # LinearConstraint(A, lb, ub)
    # We have A_ub @ x <= b_ub, which is equivalent to A_ub @ x >= -inf and A_ub @ x <= b_ub.
    # So lb = -np.inf * ones(num_constraints), ub = b_ub.
    
    cons = LinearConstraint(A_ub, lb=-np.inf * np.ones(num_constraints), ub=b_ub)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=cons,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": deadline - time.time() - 1.0, "mip_rel_gap": 0.0}
        )
    except Exception as e:
        # Fallback to greedy if solver fails or times out hard
        pass
    else:
        if res.status == 0:  # Optimal
            # Extract solution
            x = res.x
            # x should be 0 or 1 (integrality enforced)
            # Due to floating point, threshold at 0.5
            selected_indices = [i for i in range(N) if x[i] > 0.5]
            
            if selected_indices:
                result_set = [i + 1 for i in selected_indices]
                # Validate
                def _is_3ap_free(s):
                    if not s:
                        return True
                    s_set = set(s)
                    lst = list(s)
                    for i in range(len(lst)):
                        for j in range(i + 1, len(lst)):
                            a, b = lst[i], lst[j]
                            c_val = 2 * b - a
                            if c_val > b and c_val in s_set:
                                return False
                    return True
                
                if _is_3ap_free(result_set):
                    return result_set

    # Fallback: Greedy if ILP failed or returned invalid/empty
    # Greedy: pick highest weight available, remove conflicts, repeat.
    available = (1 << N) - 1
    current_set = []
    while available:
        # Find lowest set bit (highest weight since index 0 is weight 1.0)
        lsb_mask = available & -available
        idx = lsb_mask.bit_length() - 1
        
        current_set.append(idx)
        
        # Remove idx and its neighbors
        available &= ~( (1 << idx) | adj[idx] )
        
    if current_set:
        return [i + 1 for i in current_set]
    
    return [1]

if __name__ == "__main__":
    res = construct()
    print(res)
    score = sum(1.0/x for x in res)
    print(f"Score: {score}")
    print(f"Size: {len(res)}")
