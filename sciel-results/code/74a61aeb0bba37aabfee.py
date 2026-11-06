import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    deadline = time.time() + 115  # Safety margin (evaluator kills at 150s)
    
    # --- Helper functions ---
    def is_valid(A):
        """Check if adjacency matrix A represents a K2,3-free graph."""
        K = A @ A
        np.fill_diagonal(K, 0)
        return np.max(K) <= 2

    def count_edges(A):
        return int(np.sum(A)) // 2

    def can_add_edge(A, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if A[u, v] == 1:
            return True
        # For each common neighbor w of u and v, adding (u,v) increases
        # common neighbors of (u,w) by 1 and (v,w) by 1
        for w in range(n):
            if w == u or w == v:
                continue
            if A[u, w] == 1 and A[v, w] == 1:
                # (u,w) currently has cn_uw common neighbors
                cn_uw = int(np.dot(A[u, :], A[:, w]))
                if cn_uw >= 2:
                    return False
                cn_vw = int(np.dot(A[v, :], A[:, w]))
                if cn_vw >= 2:
                    return False
        return True

    def try_add_edge(A, u, v):
        if A[u, v] == 1:
            return False
        if not can_add_edge(A, u, v):
            return False
        A[u, v] = 1
        A[v, u] = 1
        return True

    def greedy_fill(A, order=None):
        """Greedily add edges in given order (or natural order)."""
        if order is None:
            order = [(u, v) for u in range(n) for v in range(u + 1, n)]
        changed = True
        while changed:
            changed = False
            for u, v in order:
                if A[u, v] == 0:
                    if try_add_edge(A, u, v):
                        changed = True
        return A

    def local_search(A, max_iters=5000):
        """Local search: try adding edges, then swapping edges to improve."""
        best_edges = count_edges(A)
        best_A = A.copy()
        
        # Phase 1: Greedy add
        A = greedy_fill(A)
        best_edges = count_edges(A)
        best_A = A.copy()
        
        # Phase 2: Swap-based local search
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try adding any missing edge
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0 and can_add_edge(A, u, v):
                        A[u, v] = 1
                        A[v, u] = 1
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                best_edges = count_edges(A)
                best_A = A.copy()
                continue
            
            # Try swapping: remove edge (u1,v1), add edge (u2,v2)
            found_swap = False
            edges_list = [(u, v) for u in range(n) for v in range(u+1, n) if A[u,v]==1]
            missing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if A[u,v]==0]
            
            # Sample some swaps for efficiency
            num_tries = min(100, len(edges_list) * len(missing_edges))
            for _ in range(num_tries):
                u1, v1 = random.choice(edges_list)
                u2, v2 = random.choice(missing_edges)
                
                # Try removing (u1,v1) and adding (u2,v2)
                A[u1, v1] = 0
                A[v1, u1] = 0
                
                if can_add_edge(A, u2, v2):
                    A[u2, v2] = 1
                    A[v2, u2] = 1
                    new_edges = count_edges(A)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_A = A.copy()
                        found_swap = True
                    else:
                        # Revert
                        A[u2, v2] = 0
                        A[v2, u2] = 0
                        A[u1, v1] = 1
                        A[v1, u1] = 1
                else:
                    # Revert
                    A[u1, v1] = 1
                    A[v1, u1] = 1
            
            if found_swap:
                continue
            
            # If no improvement, try 2-opt style: remove 2 edges, add 2 edges
            if iteration % 10 == 5:
                found_2swap = False
                for _ in range(30):
                    e1 = random.choice(edges_list)
                    e2 = random.choice(edges_list)
                    if e1 == e2:
                        continue
                    m1 = random.choice(missing_edges)
                    m2 = random.choice(missing_edges)
                    if m1 == m2:
                        continue
                    
                    # Remove e1, e2
                    A[e1[0], e1[1]] = 0
                    A[e1[1], e1[0]] = 0
                    A[e2[0], e2[1]] = 0
                    A[e2[1], e2[0]] = 0
                    
                    # Try adding m1, m2
                    if can_add_edge(A, m1[0], m1[1]):
                        A[m1[0], m1[1]] = 1
                        A[m1[1], m1[0]] = 1
                        if can_add_edge(A, m2[0], m2[1]):
                            A[m2[0], m2[1]] = 1
                            A[m2[1], m2[0]] = 1
                            new_edges = count_edges(A)
                            if new_edges > best_edges:
                                best_edges = new_edges
                                best_A = A.copy()
                                found_2swap = True
                            else:
                                A[m2[0], m2[1]] = 0
                                A[m2[1], m2[0]] = 0
                                A[m1[0], m1[1]] = 0
                                A[m1[1], m1[0]] = 0
                        else:
                            A[m1[0], m1[1]] = 0
                            A[m1[1], m1[0]] = 0
                    else:
                        pass
                    
                    # Revert removals
                    A[e1[0], e1[1]] = 1
                    A[e1[1], e1[0]] = 1
                    A[e2[0], e2[1]] = 1
                    A[e2[1], e2[0]] = 1
                    
                    if found_2swap:
                        break
                
                if found_2swap:
                    continue
        
        return best_A

    # --- MILP Formulation ---
    def solve_milp():
        """Solve the K2,3-free max edge problem via MILP."""
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_pairs = len(pairs)
        pair_index = {p: idx for idx, p in enumerate(pairs)}
        
        # y variables: for each pair (i,j) and each k != i,j
        y_vars = []
        for p, (i, j) in enumerate(pairs):
            for k in range(n):
                if k != i and k != j:
                    x_ik_idx = pair_index[(min(i,k), max(i,k))]
                    x_jk_idx = pair_index[(min(j,k), max(j,k))]
                    y_vars.append((p, k, x_ik_idx, x_jk_idx))
        
        num_y = len(y_vars)
        total_vars = num_pairs + num_y
        
        # Objective: maximize sum x => minimize -sum x
        c = np.zeros(total_vars)
        c[:num_pairs] = -1.0
        
        # Bounds and integrality
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        integrality = np.ones(total_vars)
        
        # Constraints:
        # For each y: y <= x_ik, y <= x_jk, -y + x_ik + x_jk <= 1
        # For each pair p: sum_k y_p,k <= 2
        num_constraints = 3 * num_y + num_pairs
        A_mat = np.zeros((num_constraints, total_vars))
        b_ub = np.zeros(num_constraints)
        
        ci = 0
        for y_idx, (p, k, x_ik_idx, x_jk_idx) in enumerate(y_vars):
            gy = num_pairs + y_idx
            # y - x_ik <= 0
            A_mat[ci, gy] = 1
            A_mat[ci, x_ik_idx] = -1
            b_ub[ci] = 0
            ci += 1
            # y - x_jk <= 0
            A_mat[ci, gy] = 1
            A_mat[ci, x_jk_idx] = -1
            b_ub[ci] = 0
            ci += 1
            # -y + x_ik + x_jk <= 1
            A_mat[ci, gy] = -1
            A_mat[ci, x_ik_idx] = 1
            A_mat[ci, x_jk_idx] = 1
            b_ub[ci] = 1
            ci += 1
        
        for p in range(num_pairs):
            for y_idx, (p_curr, k, _, _) in enumerate(y_vars):
                if p_curr == p:
                    A_mat[ci, num_pairs + y_idx] = 1
            b_ub[ci] = 2
            ci += 1
        
        lin_cons = LinearConstraint(A_mat, -np.inf, b_ub)
        bounds = Bounds(lb, ub)
        
        res = milp(c, constraints=lin_cons, integrality=integrality, bounds=bounds,
                   options={"time_limit": 60})
        
        if res.success and res.x is not None:
            A = np.zeros((n, n), dtype=int)
            for i in range(num_pairs):
                u, v = pairs[i]
                if res.x[i] > 0.5:
                    A[u, v] = 1
                    A[v, u] = 1
            if is_valid(A):
                return A
        return None

    # --- Main algorithm ---
    best_A = None
    best_edges = 0
    
    # Step 1: Try MILP
    if time.time() < deadline - 30:
        try:
            A_milp = solve_milp()
            if A_milp is not None:
                best_A = A_milp
                best_edges = count_edges(A_milp)
        except Exception as e:
            pass
    
    # Step 2: Heuristic initialization + local search
    if time.time() < deadline - 15:
        # Multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random structure
            A = np.zeros((n, n), dtype=int)
            
            # Add some random edges as seed
            random.seed(42 + restart)
            for _ in range(30 + restart * 3):
                u, v = random.sample(range(n), 2)
                try_add_edge(A, u, v)
            
            # Greedy fill
            A = greedy_fill(A)
            
            # Local search
            A = local_search(A, max_iters=3000)
            
            edges = count_edges(A)
            if edges > best_edges:
                best_edges = edges
                best_A = A.copy()
    
    # Step 3: Final polish if time remains
    if best_A is not None and time.time() < deadline - 5:
        A_final = local_search(best_A, max_iters=2000)
        edges = count_edges(A_final)
        if edges > best_edges:
            best_edges = edges
            best_A = A_final
    
    # Fallback: ensure we have a valid graph
    if best_A is None:
        A = np.zeros((n, n), dtype=int)
        # Add edges of a 5-clique decomposition
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        A = greedy_fill(A)
        best_A = A
        best_edges = count_edges(A)
    
    # Convert to edge list
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v]:
                edges_list.append([u, v])
    
    return edges_list
