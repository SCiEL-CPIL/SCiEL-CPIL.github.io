"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    """
    Solves the QAP instance using Linearized Block Coordinate Descent (BCD)
    and 2-swap local search.
    
    Strategy:
    1. Initialize with spectral embedding + Hungarian matching.
    2. Iteratively refine using BCD:
       - For each facility i, fix all other assignments.
       - Compute the linear cost for assigning facility i to location j:
         Cost_ij = sum_{k != i} A[i,k] B[j, p[k]]
       - Solve the 1D assignment (argmin) to update p[i].
       - This is equivalent to solving a sequence of 1x1 assignment problems,
         but we can batch them or just do it sequentially. 
         Actually, standard BCD for QAP usually updates one variable at a time.
         However, a stronger move is to update a subset or use the full 
         linearization if we treat it as a relaxation.
         
         Let's stick to single-facility updates (coordinate descent) as it's 
         guaranteed to not increase cost (if we pick the best j).
         Wait, if we change p[i], the cost changes by:
         Delta = sum_{k != i} A[i,k] (B[new_j, p[k]] - B[old_j, p[k]])
         This is O(n) per swap check. Finding the best new_j is O(n).
         Total O(n^2) per full pass.
         
         To make this "structurally different" and potentially stronger, 
         we can use the "Linearized Assignment" approach where we solve 
         the full Hungarian problem on a matrix C derived from the current 
         permutation, but that's expensive (O(n^3)). 
         
         A middle ground: 
         1. Spectral Init.
         2. Local Search (2-swap) to convergence.
         3. If time remains, perform "Kicks" (random swaps) and restart local search.
         
         The current program already does this. The prompt asks for a DIFFERENT strategy.
         Let's try **Tabu Search** with a dynamic tabu list, which is a different 
         metaheuristic family than simple greedy + random kicks.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0
    
    a_c = np.ascontiguousarray(a, dtype=np.float64)
    b_c = np.ascontiguousarray(b, dtype=np.float64)
    
    # 1. Spectral Initialization
    try:
        k = min(10, n)
        w_a, v_a = np.linalg.eigh(a_c)
        idx_a = np.argsort(w_a)[::-1]
        v_a_top = v_a[:, idx_a[:k]]
        
        w_b, v_b = np.linalg.eigh(b_c)
        idx_b = np.argsort(w_b)[::-1]
        v_b_top = v_b[:, idx_b[:k]]
        
        M = v_a_top @ v_b_top.T
        row_ind, col_ind = linear_sum_assignment(-M)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
    except Exception:
        p = np.random.permutation(n)
    
    def calc_cost(p):
        return int(np.sum(a_c * b_c[p][:, p]))
    
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Pre-compute row sums for faster delta? No, delta_swap is O(n).
    
    def delta_swap(p, i, j):
        x = p[i]
        y = p[j]
        diff_a = a_c[i, :] - a_c[j, :]
        diff_b = b_c[y, p] - b_c[x, p]
        total = np.dot(diff_a, diff_b)
        term_i = (a_c[i, i] - a_c[j, i]) * (b_c[y, x] - b_c[x, x])
        term_j = (a_c[i, j] - a_c[j, j]) * (b_c[y, y] - b_c[x, y])
        return total - term_i - term_j

    # Tabu Search Implementation
    rng = np.random.default_rng(42)
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    tabu_tenure = 20  # Number of iterations to keep a move tabu
    iteration = 0
    
    while time.time() < deadline:
        # Check time more frequently
        if time.time() > deadline - 2.0:
            break
            
        # Find best non-tabu improving move
        # To avoid O(n^2) full scan every time which might be slow if n=150 (22500 swaps),
        # we can sample or do a full scan. 22500 * O(150) dot products = 3.3M ops per pass.
        # In NumPy, vectorizing the delta calculation for all pairs is hard without loops.
        # Let's stick to a smarter local search: 
        # We will perform a full pass of steepest descent (find best swap) 
        # but optimized.
        
        # Actually, for n=150, O(n^2) swaps is 11250 unique pairs.
        # Each delta is a dot product of length 150.
        # 11250 * 150 = 1.7M multiplications. This is very fast in NumPy if vectorized?
        # But we can't easily vectorize the index access p[v] for all i,j simultaneously 
        # without a huge intermediate array.
        
        # Let's use a "Greedy First-Improvement" with random ordering to be faster 
        # and allow more iterations, combined with Tabu to avoid cycles.
        
        # Randomize facility order
        order = rng.permutation(n)
        
        found_improvement = False
        for i in order:
            if time.time() > deadline - 2.0:
                break
                
            best_j = -1
            best_delta = 0
            
            # Try swapping i with all j > i (to avoid duplicates in this pass)
            # Or just all j != i
            for j in order:
                if i == j:
                    continue
                    
                # Check tabu
                key = (min(i, j), max(i, j))
                if key in tabu_list and tabu_list[key] > iteration:
                    continue
                    
                d = delta_swap(p, i, j)
                
                if d < best_delta:
                    best_delta = d
                    best_j = j
            
            if best_j != -1:
                # Perform swap
                i_swap = i
                j_swap = best_j
                p[i_swap], p[j_swap] = p[j_swap], p[i_swap]
                current_cost += best_delta
                found_improvement = True
                
                # Add to tabu list
                key = (min(i_swap, j_swap), max(i_swap, j_swap))
                tabu_list[key] = iteration + tabu_tenure
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                # No improving move for this i in this pass
                pass
        
        iteration += 1
        
        # If no improvement found in a full pass, we are in a local minimum.
        if not found_improvement:
            # Perturbation: Random 3-swap
            indices = rng.choice(n, size=3, replace=False)
            # Swap 0-1, 1-2, 0-2? Just random swaps
            p[indices[0]], p[indices[1]] = p[indices[1]], p[indices[0]]
            p[indices[1]], p[indices[2]] = p[indices[2]], p[indices[1]]
            current_cost = calc_cost(p)
            
            # Clear tabu list to allow escape
            tabu_list.clear()
            iteration = 0
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
