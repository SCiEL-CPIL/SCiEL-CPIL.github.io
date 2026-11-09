import time
import random
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

START_TIME = time.time()
TIME_LIMIT = 110.0

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n = 11
    num_colorings = 1 << n
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute edge masks: edge_masks[i] is a bitmask of colorings invalidated by all_edges[i]
    edge_masks = []
    for e in all_edges:
        v1, v2, v3, v4, v5 = e
        mask = 0
        for c in range(num_colorings):
            b1 = (c >> v1) & 1
            b2 = (c >> v2) & 1
            b3 = (c >> v3) & 1
            b4 = (c >> v4) & 1
            b5 = (c >> v5) & 1
            if b1 == b2 == b3 == b4 == b5:
                mask |= (1 << c)
        edge_masks.append(mask)
    
    # Try ILP first
    try:
        # Build constraint matrix A (num_colorings x num_edges)
        # A[c, e] = 1 if edge e invalidates coloring c
        # We can build this by iterating edges and setting bits, then converting to sparse matrix
        # Or iterate colorings and edges.
        # Iterating edges is better for building rows if we use bitmask logic?
        # Actually, let's build the sparse matrix directly.
        
        rows = []
        cols = []
        data = []
        
        # For each edge e, find which colorings it covers.
        # We already have edge_masks.
        # We need to add 1 to A[c, e] for each set bit c in edge_masks[e].
        for e_idx, mask in enumerate(edge_masks):
            c = 0
            m = mask
            while m:
                if m & 1:
                    rows.append(c)
                    cols.append(e_idx)
                    data.append(1)
                c += 1
                m >>= 1
        
        A = csr_matrix((data, (rows, cols)), shape=(num_colorings, num_edges))
        
        # Objective: minimize sum(x_e)
        c_obj = [1] * num_edges
        
        # Constraints: A x >= 1
        constraint = LinearConstraint(A, lb=[1] * num_colorings, ub=[1] * num_colorings)
        
        # Bounds: 0 <= x_e <= 1
        bounds = Bounds(lb=[0] * num_edges, ub=[1] * num_edges)
        
        # Integrality: all variables are binary
        integrality = [1] * num_edges
        
        # Solve
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x = res.x
            selected_indices = [i for i in range(num_edges) if x[i] > 0.5]
            
            # Verify coverage
            covered = 0
            for idx in selected_indices:
                covered |= edge_masks[idx]
            
            if covered == (1 << num_colorings) - 1:
                final_edges = [list(all_edges[idx]) for idx in selected_indices]
                return {'n': n, 'edges': final_edges}
            else:
                # Fallback if ILP result is invalid (should not happen)
                pass
                
    except Exception as e:
        pass
        
    # Fallback to Greedy if ILP fails
    best_indices = None
    best_len = float('inf')
    
    for trial in range(50):
        if time_left() < 5:
            break
            
        uncovered = (1 << num_colorings) - 1
        selected_indices = []
        
        indices = list(range(len(all_edges)))
        if trial > 0:
            random.shuffle(indices)
            
        while uncovered:
            best_gain = -1
            best_idx = -1
            
            for idx in indices:
                em = edge_masks[idx]
                gain = (em & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                elif gain == best_gain and gain > 0:
                    if trial > 0 and random.random() < 0.5:
                        best_idx = idx
            
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            
            if len(selected_indices) >= best_len:
                break
        
        if not uncovered and len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_indices = selected_indices[:]
            
            if best_len <= 29: # Theoretical min
                break

    if best_indices is None:
        return {'n': n, 'edges': [list(e) for e in all_edges[:100]]}
        
    # Local Search on Greedy result
    current_indices = best_indices[:]
    improved = True
    while improved and time_left() > 5:
        improved = False
        total_coverage = 0
        for idx in current_indices:
            total_coverage |= edge_masks[idx]
            
        for i in range(len(current_indices)):
            if time_left() < 5:
                break
            cov_without = 0
            for j in range(len(current_indices)):
                if i != j:
                    cov_without |= edge_masks[current_indices[j]]
                    
            if cov_without == total_coverage:
                current_indices.pop(i)
                improved = True
                break
                
    final_edges = [list(all_edges[idx]) for idx in current_indices]
    return {'n': n, 'edges': final_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
