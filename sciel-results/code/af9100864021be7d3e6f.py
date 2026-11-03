"""
Strategy: Simulated Annealing with K-opt Block Moves and Spectral Initialization for QAP.
1. Initialize using the Fiedler vector (second eigenvector) of the flow matrix A to create a structurally informed 1D ordering, then map to locations based on distance sums.
2. Use Simulated Annealing to explore the search space.
3. Moves include:
   - 2-opt: Swap two facilities.
   - K-opt: Reverse a contiguous segment of the permutation (Or-opt style) or swap two contiguous segments.
   - This allows escaping local optima that simple 2-opt gets stuck in.
4. Efficient delta calculation for 2-opt. For K-opt, recompute cost if segment is small or use approximate delta, but given N=256, we can afford O(N^2) cost check for occasional large moves or efficient O(N*K) for block swaps if derived, but let's stick to robust 2-opt + random segment reversals with full cost check for the segment part or just accept the cost of a full O(N^2) re-evaluation for the perturbation step if it's rare. 
   Actually, for N=256, full cost is 256^2 = 65k ops. Very fast in NumPy.
   We can use full cost evaluation for any move that is not a simple 2-opt swap, or even for all moves if we batch them, but delta is faster.
   Let's use delta for 2-opt and full cost for segment reversals (Or-opt) since deriving delta for segment reversal is complex.
   Frequency: 80% 2-opt, 20% Or-opt (reverse segment).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    # p is array of locations for each facility
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym):
    """
    Compute the change in cost when swapping facilities at positions i and j.
    p[i] is location of facility i.
    a_sym = a + a.T
    """
    x = i
    y = j
    Lx = p[x]
    Ly = p[y]
    
    if Lx == Ly:
        return 0
        
    # diff_b[k] = b[Ly, p[k]] - b[Lx, p[k]]
    diff_b = b[Ly, p] - b[Lx, p]
    
    # coeff[k] = a_sym[x, k] - a_sym[y, k]
    coeff = a_sym[x, :] - a_sym[y, :]
    
    delta = np.dot(coeff, diff_b)
    return delta

def _spectral_init(a, b):
    """
    Initialize permutation using spectral ordering.
    1. Find Fiedler vector of A (or -A to maximize spread? We want to minimize sum a[i,j]b[pi,pj].
       If A has positive entries, high flow should go to low distance.
       Spectral embedding of A: Eigenvectors of A give a layout where high flow is close?
       Actually, usually we embed the graph defined by A.
       Let's use the second smallest eigenvector of the Laplacian of A? Or just eigenvector of A.
       Standard approach: Use eigenvector of A to order facilities.
       Then order locations by their distance sum (central first).
    """
    n = len(a)
    # Use eigendecomposition of A
    # For large N, scipy.sparse.linalg.eigsh is better, but N=256 is small enough for dense
    # However, A might be non-symmetric. Use (A+A.T)/2 for symmetry.
    a_sym = (a + a.T) / 2.0
    
    try:
        # Get eigenvalues and eigenvectors
        # We want the eigenvector corresponding to the smallest eigenvalue (or second smallest if Laplacian)
        # For QAP initialization, using the eigenvector of the flow matrix itself is common.
        # Let's use the eigenvector associated with the smallest eigenvalue of the Laplacian of the flow graph?
        # Or simply the Fiedler vector of the graph defined by A.
        
        # Simple heuristic: Project facilities onto the principal axis of A.
        # Let's use the second smallest eigenvector of the Laplacian of A_sym.
        # L = D - A_sym
        deg = a_sym.sum(axis=1)
        L = np.diag(deg) - a_sym
        
        # Use np.linalg.eigh for symmetric L
        vals, vecs = np.linalg.eigh(L)
        
        # Fiedler vector is usually the second smallest eigenvalue (index 1)
        # If the graph is connected, index 0 is 0.
        # If not, we might need to be careful.
        # Let's take the eigenvector with the second smallest eigenvalue.
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler_vec = vecs[:, idx[1]]
        
        # Order facilities by their projection on the Fiedler vector
        fac_order = np.argsort(fiedler_vec, kind="stable")
        
    except Exception:
        # Fallback to random or row sum
        fac_order = np.random.permutation(n)
        
    # Order locations by distance sum (ascending: central first)
    b_row_sums = b.sum(axis=1)
    loc_order = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    # Assign facility fac_order[k] to location loc_order[k]
    for k in range(n):
        p[fac_order[k]] = loc_order[k]
        
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    a_sym = a + a.T
    
    # 1. Initialization
    p = _spectral_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    # Start temperature high enough to accept bad moves, cool down
    # Initial cost is likely high. Let's estimate typical delta magnitude.
    # Random swap delta magnitude for N=256 might be in the range of 10^4 - 10^6?
    # Best known cost ~ 4.5e7.
    # Let's set T0 = current_cost / 10.0 or similar.
    T = current_cost / 10.0
    T_min = 1.0
    cooling_rate = 0.999
    
    # Track no-improvement count to reset temperature if stuck? 
    # Or just let it cool.
    
    steps = 0
    last_improve_time = start_time
    
    while time.time() - start_time < time_limit:
        steps += 1
        
        # Generate move
        if random.random() < 0.8:
            # 2-opt swap
            i, j = random.sample(range(n), 2)
            delta = _swap_delta(a, b, p, i, j, a_sym)
            if delta < 0:
                # Accept
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    last_improve_time = time.time()
            else:
                # Metropolis criterion
                if T > T_min:
                    # Accept with probability exp(-delta/T)
                    # Avoid overflow/underflow
                    if delta < 100 * T: # Heuristic check
                        prob = np.exp(-delta / T)
                        if random.random() < prob:
                            p[i], p[j] = p[j], p[i]
                            current_cost += delta
                # else reject
        else:
            # Or-opt: Reverse a segment
            # Pick a random segment length k (2 to 10)
            k = random.randint(2, 10)
            if k < n:
                start = random.randint(0, n - k)
                end = start + k - 1
                # Calculate cost change for reversing segment [start, end]
                # This is more complex. For simplicity and speed, 
                # we can just do the reversal and calculate the new cost if the segment is small?
                # No, _qap_cost is O(N^2). 
                # Frequency of Or-opt is 20%. 
                # If we do full cost calc for 20% of moves, it might be slow.
                # Let's optimize: Only do Or-opt if we are stuck (no improvement for a while) or early on.
                # Or, implement a fast delta for segment reversal?
                # Delta for reversing segment S = [s, e]:
                # The relative order of facilities in S is reversed.
                # Interactions within S: a[i,j] b[p[i], p[j]] -> a[i,j] b[p[j], p[i]] (if i,j in S and i<j, they swap positions relative to each other? No, the set of locations assigned to S is the same, but the mapping is reversed).
                # Actually, if we reverse the segment of the permutation p, the locations assigned to the facilities in S are the same set, but permuted.
                # Let P_S be the sub-permutation.
                # This is getting complicated. 
                # Alternative: Use 3-opt (move one facility to another position) or just stick to 2-opt but with higher temperature?
                # Given the time constraint and N=256, let's try to implement a simple "block swap" delta or just use 2-opt with aggressive cooling and restarts.
                # Let's stick to 2-opt for speed, but add a "kick" mechanism: if no improvement for X seconds, do 10 random swaps and restart cooling.
                
                # For now, let's just do another 2-opt or a random restart of the segment.
                # Let's try a "reverse segment" move with full cost check, but limit its frequency to 5% to keep speed up.
                if random.random() < 0.05:
                    # Do the reversal
                    p[start:end+1] = p[start:end+1][::-1]
                    new_cost = _qap_cost(a, b, p)
                    delta = new_cost - current_cost
                    if delta < 0:
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
                            last_improve_time = time.time()
                    else:
                        # Accept with prob
                        if T > T_min and delta < 100 * T:
                            prob = np.exp(-delta / T)
                            if random.random() < prob:
                                current_cost = new_cost
                            else:
                                # Revert
                                p[start:end+1] = p[start:end+1][::-1]
                        else:
                            # Revert
                            p[start:end+1] = p[start:end+1][::-1]
        
        # Cooling
        if T > T_min:
            T *= cooling_rate
        
        # Check for stagnation and restart cooling if stuck
        if time.time() - last_improve_time > 5.0:
            # Reset temperature to escape local optimum
            T = current_cost / 10.0
            # Optional: Perturb current solution
            for _ in range(5):
                i, j = random.sample(range(n), 2)
                p[i], p[j] = p[j], p[i]
            current_cost = _qap_cost(a, b, p)
            # If this perturbation made it much worse, we might want to revert to best_p?
            # SA usually keeps the current state. But if we jumped to a bad region, it's slow to get back.
            # Let's restart from best_p if the perturbation increased cost by more than 10%.
            if current_cost > best_cost * 1.1:
                p = best_p.copy()
                current_cost = best_cost
            last_improve_time = time.time()

    return best_p.tolist()
# EVOLVE-BLOCK-END
