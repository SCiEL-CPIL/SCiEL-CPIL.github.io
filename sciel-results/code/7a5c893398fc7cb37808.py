import numpy as np
import time
import itertools

def has_k4(adj, n):
    """Returns True if graph has a K4."""
    # adj is a list of int masks or a numpy array? 
    # For speed in the verification step, use bit masks or numpy.
    # Let's assume adj is a numpy boolean array for this function, or convert.
    # But the main search uses numpy. The final return needs list of edges.
    # This function is used for final verification.
    
    # Convert to numpy bool array if not already
    A = np.zeros((n, n), dtype=bool)
    # If adj is a list of masks:
    # It's better to pass the numpy matrix directly to the verifier to avoid conversion overhead if called often.
    # But construct() calls it once.
    
    # Efficient K4 check using numpy
    # K4 exists if there is a 4-clique.
    # We can iterate over pairs and check common neighbors.
    for u in range(n):
        for v in range(u + 1, n):
            if A[u, v]:
                common = np.logical_and(A[u], A[v])
                common[u] = False
                common[v] = False
                if common.sum() >= 2:
                    # Check if any two common neighbors are connected
                    # This is O(n^2) per pair, which is slow for n=35? 
                    # 35*35/2 = 612 pairs.
                    # If common sum is 2, check the two.
                    # If common sum is large, check quickly.
                    # Actually, if common sum >= 2, we just need one edge among them.
                    idx = np.nonzero(common)[0]
                    # Check subgraph induced by idx
                    for i in range(len(idx)):
                        for j in range(i + 1, len(idx)):
                            if A[idx[i], idx[j]]:
                                return True
    return False

def has_ind_6(adj_np, n):
    """Returns True if graph has an independent set of size 6."""
    # Independent set in G is Clique in G'
    # G' = J - I - G
    comp = np.ones((n, n), dtype=bool)
    np.fill_diagonal(comp, False)
    comp = comp & ~adj_np
    
    def dfs(cands_mask, depth):
        if depth == 0:
            return True
        if cands_mask.sum() < depth:
            return False
        
        # Heuristic: pick vertex with max degree in induced subgraph? 
        # For clique finding, min degree in complement? 
        # In the subgraph on cands, we want a clique of size 'depth' in comp.
        # Greedy: pick vertex with max degree in comp induced by cands.
        v = -1
        max_deg = -1
        # cands_mask is a boolean vector
        for u in np.nonzero(cands_mask)[0]:
            deg = (comp[u] & cands_mask).sum()
            if deg > max_deg:
                max_deg = deg
                v = u
        
        if v == -1:
            return False
            
        # Branch 1: Include v
        new_cands = cands_mask.copy()
        new_cands[v] = False
        # Neighbors of v in comp
        if dfs(new_cands & comp[v], depth - 1):
            return True
        
        # Branch 2: Exclude v
        if dfs(new_cands, depth):
            return True
            
        return False
        
    return dfs(np.ones(n, dtype=bool), 6)

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # We try n from 35 down to 25
    for n in [35, 34, 33, 32, 31, 30]:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random graph
        # Use numpy for speed
        A = np.zeros((n, n), dtype=bool)
        # Set random edges
        # To make it symmetric:
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < 0.4:
                    A[i, j] = True
                    A[j, i] = True
                    
        # Simulated Annealing
        # Surrogate: Tr(A^4) + Tr(Comp^6)
        # We want to minimize this.
        
        def get_surrogate(A):
            # A^4
            A2 = A @ A
            A4 = A2 @ A2
            tr_A4 = np.trace(A4)
            
            # Comp
            J = np.ones((n, n), dtype=bool)
            np.fill_diagonal(J, False)
            Comp = J & ~A
            C2 = Comp @ Comp
            C4 = C2 @ C2
            C6 = C4 @ C2
            tr_C6 = np.trace(C6)
            
            return tr_A4 + tr_C6

        current_A = A
        current_score = get_surrogate(current_A)
        
        best_A = current_A.copy()
        best_score = current_score
        
        temp = 1.0
        step = 0
        max_steps = 50000  # Adjust based on speed
        
        while step < max_steps:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Pick random edge to flip
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            # Flip edge
            old_val = current_A[i, j]
            current_A[i, j] = not old_val
            current_A[j, i] = not old_val
            
            new_score = get_surrogate(current_A)
            
            # Accept if better, or with probability exp((old-new)/temp)
            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = current_A.copy()
                    # If best_score is 0, we have a candidate
                    if best_score == 0:
                        # Verify exactly
                        if not has_k4(best_A, n) and not has_ind_6(best_A, n):
                            # Found valid graph
                            edges = []
                            for u in range(n):
                                for v in range(u + 1, n):
                                    if best_A[u, v]:
                                        edges.append([u, v])
                            return {'n': n, 'edges': edges}
            else:
                # Reject
                current_A[i, j] = old_val
                current_A[j, i] = old_val
            
            step += 1
            # Cooling
            temp *= 0.999
            
        # If no solution found for this n, try next n
        # But we should restart with a new random graph for this n? 
        # Given time constraints, we might just drop n.
        
    # Fallback: return a trivial valid graph
    # For n=1, no K4, no Ind 6.
    return {'n': 1, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(f"Result: n={res['n']}, edges={len(res['edges'])}")
