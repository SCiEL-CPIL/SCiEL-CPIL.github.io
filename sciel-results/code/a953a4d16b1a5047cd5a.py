import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    def get_seed_graph():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges
    
    best_n, best_edges = get_seed_graph()
    
    # Try to find larger graphs
    for target_n in [42, 41, 40, 39, 38]:
        if time.time() > deadline:
            break
        
        time_budget = deadline - time.time()
        if time_budget < 10:
            break
        
        found = solve_for_n(target_n, time_budget)
        if found:
            n, edges = found
            if n > best_n:
                best_n, best_edges = n, edges
                if n >= 42:
                    break
    
    return best_n, best_edges

def solve_for_n(n, time_budget):
    """
    Use simulated annealing with a smooth objective function.
    The objective counts the number of "potential" K5 and I5 violations:
    - For each vertex v, count 4-cliques in its neighborhood (each would form K5 with v)
    - For each vertex v, count 4-independent sets in its non-neighborhood (each would form I5 with v)
    
    We minimize this count. When it reaches 0, we have a valid graph.
    """
    rng = random.Random(12345 + n)
    np_rng = np.random.RandomState(42 + n)
    
    # Initialize adjacency matrix from Paley-like structure extended to n
    # Start with a random graph biased toward the Paley structure for first 37 vertices
    A = np.zeros((n, n), dtype=np.uint8)
    
    # For the first 37 vertices, use Paley graph
    q = 37
    residues = set((x * x) % q for x in range(1, q))
    for i in range(min(q, n)):
        for j in range(i + 1, min(q, n)):
            if (j - i) % q in residues:
                A[i, j] = 1
                A[j, i] = 1
    
    # For vertices beyond 37, connect randomly
    for i in range(q, n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                A[i, j] = 1
                A[j, i] = 1
        for j in range(i):
            if j < q:
                # Connect to Paley vertices with some probability
                if rng.random() < 0.5:
                    A[i, j] = 1
                    A[j, i] = 1
    
    # Ensure symmetry
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    
    def compute_objective(A):
        """
        Count potential K5 and I5 violations.
        For each vertex v:
          - Count 4-cliques in N(v): these would form K5 with v
          - Count 4-independent sets in non-N(v): these would form I5 with v
        
        We use a matrix-based approach for efficiency.
        """
        total = 0
        n = A.shape[0]
        
        for v in range(n):
            # Get neighbors and non-neighbors of v
            neighbors = np.where(A[v] == 1)[0]
            non_neighbors = np.where(A[v] == 0)[0]
            non_neighbors = non_neighbors[non_neighbors != v]
            
            # Count 4-cliques in the subgraph induced by neighbors
            if len(neighbors) >= 4:
                # Build subgraph on neighbors
                sub = A[np.ix_(neighbors, neighbors)]
                # Count 4-cliques: check all combinations of 4
                for combo in combinations(range(len(neighbors)), 4):
                    # Check if all pairs in combo are connected
                    is_clique = True
                    for i in range(4):
                        for j in range(i + 1, 4):
                            if sub[combo[i], combo[j]] == 0:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        total += 1
            
            # Count 4-independent sets in the subgraph induced by non-neighbors
            if len(non_neighbors) >= 4:
                sub = A[np.ix_(non_neighbors, non_neighbors)]
                # A 4-independent set means no edges among the 4
                for combo in combinations(range(len(non_neighbors)), 4):
                    is_indep = True
                    for i in range(4):
                        for j in range(i + 1, 4):
                            if sub[combo[i], combo[j]] == 1:
                                is_indep = False
                                break
                        if not is_indep:
                            break
                    if is_indep:
                        total += 1
        
        return total
    
    def is_valid(A):
        """Check if graph has no K5 and no I5."""
        n = A.shape[0]
        for combo in combinations(range(n), 5):
            # Check if K5
            is_k5 = True
            is_i5 = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if A[combo[i], combo[j]] == 1:
                        is_i5 = False
                    else:
                        is_k5 = False
                    if not is_k5 and not is_i5:
                        break
                if not is_k5 and not is_i5:
                    break
            if is_k5 or is_i5:
                return False
        return True
    
    # Simulated annealing
    current_obj = compute_objective(A)
    best_A = A.copy()
    best_obj = current_obj
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    
    while time.time() < deadline - 5:
        if current_obj == 0:
            # Verify it's truly valid
            if is_valid(A):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if A[i, j] == 1:
                            edges.append((i, j))
                return n, edges
            break
        
        # Pick a random edge to flip
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        # Compute delta objective
        # Flipping edge (u,v) affects:
        # - For each other vertex w, the 4-clique/I5 counts involving {u,v,w,x,y}
        # This is complex, so we just recompute locally for affected vertices
        
        old_A = A[u, v]
        A[u, v] = 1 - old_A
        A[v, u] = 1 - old_A
        
        new_obj = compute_objective(A)
        delta = new_obj - current_obj
        
        if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = A.copy()
        else:
            # Revert
            A[u, v] = old_A
            A[v, u] = old_A
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Check best solution
    if best_obj == 0:
        if is_valid(best_A):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_A[i, j] == 1:
                        edges.append((i, j))
            return n, edges
    
    return None

def find_one_violation(n, edges_set):
    """Fallback: find one K5 or I5 violation."""
    for combo in combinations(range(n), 5):
        is_k5 = True
        is_ik5 = True
        for i in range(5):
            for j in range(i + 1, 5):
                u, v = combo[i], combo[j]
                if u > v:
                    u, v = v, u
                if (u, v) in edges_set:
                    is_ik5 = False
                else:
                    is_k5 = False
                if not is_k5 and not is_ik5:
                    break
            if not is_k5 and not is_ik5:
                break
        if is_k5:
            return combo, 'K5'
        if is_ik5:
            return combo, 'IK5'
    return None, None
