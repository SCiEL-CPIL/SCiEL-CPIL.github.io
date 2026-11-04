"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _local_search_2opt(a, b, p, time_limit, start_time):
    """
    Fast vectorized 2-opt local search for QAP.
    Uses D = b[p,p] and A_sym = a + a.T for efficient delta calculation.
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation for swapping i with each j in js
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction terms
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = A_sym[i, js] - A_sym[js, js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _vns_search(a, b, p, time_limit, start_time):
    """
    Variable Neighborhood Search (VNS) for QAP.
    Systematically expands neighborhood size: 2-opt, then 3-opt (relocation + cyclic shift).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    def _apply_swap(p, D, i, j):
        """Swap positions i and j in p and update D."""
        p[i], p[j] = p[j], p[i]
        D[i], D[j] = D[j].copy(), D[i].copy()
        D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
    
    def _apply_relocation(p, D, i, j):
        """Move element at position i to position j (shifting elements in between)."""
        if i == j:
            return
        val = p[i]
        if i < j:
            p[i:j] = p[i+1:j+1]
            p[j] = val
        else:
            p[i+1:j+1] = p[i:j]
            p[i] = val
        
        # Update D: this is more complex. We need to rebuild D rows/cols for affected indices.
        # For simplicity and correctness, let's just rebuild D from scratch for relocation.
        # This is O(n^2) but relocation is used sparingly in VNS.
        D[:] = b[np.ix_(p, p)]
    
    def _apply_cyclic_shift(p, D, i, j):
        """Cyclically shift elements from position i to j (inclusive) by one position to the right."""
        if i >= j:
            return
        # p[i:j+1] = [p[i], p[i+1], ..., p[j]]
        # becomes [p[j], p[i], p[i+1], ..., p[j-1]]
        val = p[j]
        p[i:j] = p[i+1:j+1]
        p[j] = val
        
        # Rebuild D for simplicity
        D[:] = b[np.ix_(p, p)]
    
    # Neighborhood 1: 2-opt (swaps)
    # Neighborhood 2: 3-opt (relocations)
    # Neighborhood 3: Cyclic shifts of length 3
    
    k = 0  # neighborhood index
    max_k = 2  # 0: swap, 1: relocation, 2: cyclic shift
    
    while time.time() - start_time < time_limit:
        improved = False
        
        if k == 0:
            # 2-opt neighborhood
            p, current_cost = _local_search_2opt(a, b, p, time_limit, start_time)
            # Check if we improved from the start of this VNS iteration
            # Actually, _local_search_2opt already does a full pass.
            # We need to compare with the cost before this VNS step.
            # Let's restructure: _local_search_2opt returns the best it found.
            # If it's better than the current best, we accept and reset k.
            # But we need to track the cost before the VNS step.
            
            # Let's restructure the loop to be clearer.
            break  # This simple structure is flawed. Let's implement a proper VNS loop.
        
        # Proper VNS implementation:
        # We'll use a different structure below.
    
    # Let's implement a cleaner VNS loop.
    # We'll keep p and D, and try neighborhoods in order.
    
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    k = 0
    while time.time() - start_time < time_limit:
        improved = False
        
        if k == 0:
            # 2-opt: do a full pass of best-improvement swaps
            p_ls, cost_ls = _local_search_2opt(a, b, p, time_limit, start_time)
            if cost_ls < current_cost:
                p = p_ls
                D = b[np.ix_(p, p)].copy()
                current_cost = cost_ls
                improved = True
                k = 0  # stay in 2-opt
            else:
                k = 1  # move to next neighborhood
                
        elif k == 1:
            # Relocation neighborhood: try moving each facility to a random position
            # To keep it fast, we'll try a limited number of relocations
            for _ in range(10):  # try 10 random relocations
                if time.time() - start_time > time_limit:
                    break
                
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                
                # Save state
                p_old = p.copy()
                cost_old = current_cost
                
                # Apply relocation
                _apply_relocation(p, D, i, j)
                new_cost = int(np.sum(a * D))
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    k = 0  # reset to 2-opt
                    break
                else:
                    # Revert
                    p = p_old
                    D = b[np.ix_(p, p)].copy()
            
            if not improved:
                k = 2  # move to next neighborhood
                
        elif k == 2:
            # Cyclic shift neighborhood
            for _ in range(10):  # try 10 random cyclic shifts
                if time.time() - start_time > time_limit:
                    break
                
                i = random.randint(0, n-2)
                j = random.randint(i+1, n-1)
                
                # Save state
                p_old = p.copy()
                cost_old = current_cost
                
                # Apply cyclic shift
                _apply_cyclic_shift(p, D, i, j)
                new_cost = int(np.sum(a * D))
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    k = 0  # reset to 2-opt
                    break
                else:
                    # Revert
                    p = p_old
                    D = b[np.ix_(p, p)].copy()
            
            if not improved:
                # All neighborhoods exhausted without improvement
                # Perturb and restart VNS
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Random perturbation: swap 3 random pairs
                for _ in range(3):
                    i, j = random.sample(range(n), 2)
                    _apply_swap(p, D, i, j)
                    current_cost = int(np.sum(a * D))
                
                k = 0  # restart from 2-opt
        
        if time.time() - start_time > time_limit:
            break
    
    return p, current_cost

def _rank_matching_init(a, b, n):
    """
    Rank matching initialization: assign facilities with high flow sums to locations with low distance sums.
    """
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _eigen_init(a, b, n, k=10):
    """
    Eigenvalue-based initialization.
    """
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    try:
        import scipy.linalg as sla
        w_a, v_a = sla.eigh(A_sym, subset_by_index=[n-k, n-1])
        w_b, v_b = sla.eigh(B_sym, subset_by_index=[n-k, n-1])
    except Exception:
        w_a, v_a = np.linalg.eigh(A_sym)
        v_a = v_a[:, -k:]
        w_b, v_b = np.linalg.eigh(B_sym)
        v_b = v_b[:, -k:]
        
    # Align signs
    for i in range(k):
        if np.dot(v_a[:, i], v_b[:, i]) < 0:
            v_a[:, i] = -v_a[:, i]
            
    # Initial assignment based on sorting coordinates
    coords_a = v_a[:, 0]
    coords_b = v_b[:, 0]
    
    order_b = np.argsort(coords_b)
    order_a = np.argsort(coords_a)
    
    p = np.empty(n, dtype=int)
    p[order_a] = order_b
    
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Generate diverse initial solutions
    initial_solutions = []
    
    # 1. Rank matching
    p_rank = _rank_matching_init(a, b, n)
    initial_solutions.append(p_rank)
    
    # 2. Eigen-based
    p_eigen = _eigen_init(a, b, n, k=10)
    initial_solutions.append(p_eigen)
    
    # 3. Random
    for _ in range(2):
        initial_solutions.append(np.random.permutation(n))
    
    # Phase 1: Local search on each initial solution
    initial_time_fraction = 0.4
    for idx, p_init in enumerate(initial_solutions):
        if time.time() - start_time > time_budget * initial_time_fraction:
            break
        
        elapsed = time.time() - start_time
        remaining_initial_time = time_budget * initial_time_fraction - elapsed
        time_limit = max(1, remaining_initial_time / (len(initial_solutions) - idx))
        
        p_ls, cost_ls = _local_search_2opt(a, b, p_init, time_limit, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: VNS on the best solution found
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 10:
        return best_p.tolist()
    
    p_vns, cost_vns = _vns_search(a, b, best_p, remaining_time, start_time)
    if cost_vns < best_cost:
        best_cost = cost_vns
        best_p = p_vns.copy()
    
    # Phase 3: Final polish with 2-opt
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time > 5:
        p_final, cost_final = _local_search_2opt(a, b, best_p, remaining_time, start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
